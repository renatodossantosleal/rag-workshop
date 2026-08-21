import json
import os
import boto3
import time
from botocore.exceptions import ClientError
import pprint
from retrying import retry
import zipfile
from io import BytesIO
import warnings
import random
warnings.filterwarnings('ignore')

WORKSHOP_TAGS_MAP = {"workshop-kb": "true", "rag-workshop": "true"}
WORKSHOP_TAGS_IAM = [
    {"Key": "workshop-kb", "Value": "true"},
    {"Key": "rag-workshop", "Value": "true"},
]

DEFAULT_GENERATION_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_EVALUATION_MODEL_ID = "anthropic.claude-sonnet-4-6"
DEFAULT_GENERATION_INFERENCE_PROFILE_ID = (
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
DEFAULT_EVALUATION_INFERENCE_PROFILE_ID = "us.anthropic.claude-sonnet-4-6"

valid_generation_models = [
    DEFAULT_GENERATION_MODEL_ID,
    DEFAULT_EVALUATION_MODEL_ID,
    DEFAULT_GENERATION_INFERENCE_PROFILE_ID,
    DEFAULT_EVALUATION_INFERENCE_PROFILE_ID,
    "amazon.nova-micro-v1:0",
]


pp = pprint.PrettyPrinter(indent=2)

def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)

class BedrockStructuredKnowledgeBase:
    def __init__(
            self,
            kb_name=None,
            kb_description=None,
            workgroup_arn=None,
            secrets_arn = None,
            kbConfigParam = None,
            generation_model=os.getenv("BEDROCK_TEXT_MODEL_ID", DEFAULT_GENERATION_MODEL_ID),
            suffix=None,
    ):
        boto3_session = boto3.session.Session()
        self.region_name = boto3_session.region_name
        self.iam_client = boto3_session.client('iam')
        self.account_number = boto3.client('sts').get_caller_identity().get('Account')
        self.suffix = suffix or f'{self.region_name}-{self.account_number}'
        self.identity = boto3.client('sts').get_caller_identity()['Arn']
        self.bedrock_agent_client = boto3.client('bedrock-agent')
        credentials = boto3.Session().get_credentials()

        self.kb_name = kb_name or f"structured-knowledge-base-{self.suffix}"
        self.kb_description = kb_description or "Structures Knowledge Base"
        self.generation_model = generation_model
        self.workgroup_arn = workgroup_arn
        self.secrets_arn = secrets_arn
        self.kbConfigParam = kbConfigParam
        
        self._validate_models()
        
        self.kb_execution_role_name = f'AmazonBedrockExecutionRoleForStructuredKnowledgeBase_{self.suffix}'
        self.fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{self.suffix}'
        self.sm_policy_name = f'AmazonBedrockSecretPolicyForKnowledgeBase_{self.suffix}'
        self.rs_policy_name = f'AmazonBedrockRedshiftPolicyForKnowledgeBase_{self.suffix}'
        self.cw_log_policy_name = f'AmazonBedrockCloudWatchPolicyForKnowledgeBase_{self.suffix}'
        self.roles = [self.kb_execution_role_name]

        self.vector_store_name = f'bedrock-sample-structured-rag-{self.suffix}'
        self.index_name = f"bedrock-structured-rag-index-{self.suffix}"

        self._setup_resources()

    def _validate_models(self):
        if self.generation_model not in valid_generation_models:
            raise ValueError(f"Invalid Generation model. Your generation model should be one of {valid_generation_models}")
       

    def _setup_resources(self):

        print("========================================================================================")
        print(f"Step 1 - Creating Knowledge Base Execution Role ({self.kb_execution_role_name}) and Policies")
        
        self.bedrock_kb_execution_role = self.create_bedrock_execution_role_structured_rag()
        self.bedrock_kb_execution_role_name = self.bedrock_kb_execution_role['Role']['RoleName']

        print("========================================================================================")
        print(f"Step 2 - Creating Knowledge Base")
        self.knowledge_base, self.data_source = self.create_structured_knowledge_base()
        print("========================================================================================")

    def create_bedrock_execution_role_structured_rag(self):

        # 0. Create bedrock execution role

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com" 
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # create bedrock execution role
        try:
            bedrock_kb_execution_role = self.iam_client.create_role(
                RoleName=self.kb_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description='Amazon Bedrock Knowledge Base Execution Role for accessing redshift',
                MaxSessionDuration=3600,
                Tags=WORKSHOP_TAGS_IAM,
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            bedrock_kb_execution_role = self.iam_client.get_role(
                RoleName=self.kb_execution_role_name
            )

        # fetch arn of the role created above
        bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']

        # 1. Cretae and attach policy for foundation models
        redshift_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "RedshiftDataAPIStatementPermissions",
                    "Effect": "Allow",
                    "Action": [
                        "redshift-data:GetStatementResult",
                        "redshift-data:DescribeStatement",
                        "redshift-data:CancelStatement"
                    ],
                    "Resource": [
                        "*"
                    ],
                    "Condition": {
                    "StringEquals": {
                        "redshift-data:statement-owner-iam-userid": "${aws:userid}"
                        }
                    }
                },
                {
                "Sid": "RedshiftDataAPIExecutePermissions",
                "Effect": "Allow",
                "Action": [
                    "redshift-data:ExecuteStatement"
                ],
                "Resource": [
                    f"{self.workgroup_arn}"
                ]
            },
            # {
            #     "Sid": "RedshiftServerlessGetCredentials",
            #     "Effect": "Allow",
            #     "Action": "redshift-serverless:GetCredentials",
            #     "Resource": [
            #         f"{workgroup_arn}"
            #     ]
            # },
        
            # {
            #     "Sid": "GetSecretPermissions",
            #     "Effect": "Allow",
            #     "Action": [
            #         "secretsmanager:GetSecretValue"
            #     ],
            #     "Resource": [
            #         f"{self.secrets_arn}"
            #     ]
            # },
            
            {
                "Sid": "SqlWorkbenchAccess",
                "Effect": "Allow",
                "Action": [
                    "sqlworkbench:GetSqlRecommendations",
                    "sqlworkbench:PutSqlGenerationContext",
                    "sqlworkbench:GetSqlGenerationContext",
                    "sqlworkbench:DeleteSqlGenerationContext"
                ],
                "Resource": "*"
            },
            {
                "Sid": "KbAccess",
                "Effect": "Allow",
                "Action": [
                    "bedrock:GenerateQuery"
                ],
                "Resource": "*"
            }
            ]
        }

        if self.secrets_arn:
            redshift_policy_document['Statement'].append(
                {
                    "Sid": "GetSecretPermissions",
                    "Effect": "Allow",
                    "Action": [
                        "secretsmanager:GetSecretValue"
                    ],
                    "Resource": [
                        f"{self.secrets_arn}"
                    ]
                }
            )
        else:
            redshift_policy_document['Statement'].append(
                {
                    "Sid": "RedshiftServerlessGetCredentials",
                    "Effect": "Allow",
                    "Action": "redshift-serverless:GetCredentials",
                    "Resource": [
                        f"{self.workgroup_arn}"
                    ]
                }

            )
        
        try:
            redshift_policy = self.iam_client.create_policy(
                PolicyName=self.rs_policy_name,
                PolicyDocument=json.dumps(redshift_policy_document),
                Description='Policy for redshift workgroup',
                Tags=WORKSHOP_TAGS_IAM,
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            redshift_policy = self.iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{self.account_number}:policy/"
                f"{self.rs_policy_name}"
            )
    
        # fetch arn of this policy 
        redshift_policy_arn = redshift_policy["Policy"]["Arn"]
        
        # attach this policy to Amazon Bedrock execution role
        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=redshift_policy_arn
        )

        return bedrock_kb_execution_role
    
    def _list_knowledge_bases(self):
        summaries = []
        request = {"maxResults": 100}
        while True:
            response = self.bedrock_agent_client.list_knowledge_bases(**request)
            summaries.extend(response.get("knowledgeBaseSummaries", []))
            next_token = response.get("nextToken")
            if not next_token:
                return summaries
            request["nextToken"] = next_token

    def _list_data_sources(self, kb_id):
        summaries = []
        request = {"knowledgeBaseId": kb_id, "maxResults": 100}
        while True:
            response = self.bedrock_agent_client.list_data_sources(**request)
            summaries.extend(response.get("dataSourceSummaries", []))
            next_token = response.get("nextToken")
            if not next_token:
                return summaries
            request["nextToken"] = next_token

    def _get_knowledge_base_by_name(self):
        matching = [
            summary for summary in self._list_knowledge_bases()
            if summary.get("name") == self.kb_name
        ]
        if not matching or not matching[0].get("knowledgeBaseId"):
            raise RuntimeError(
                f"Knowledge Base {self.kb_name!r} was not visible after a conflict"
            )
        return self.bedrock_agent_client.get_knowledge_base(
            knowledgeBaseId=matching[0]["knowledgeBaseId"]
        )["knowledgeBase"]

    def _wait_for_knowledge_base_active(self, kb):
        kb_id = kb["knowledgeBaseId"]
        for _ in range(30):
            response = self.bedrock_agent_client.get_knowledge_base(
                knowledgeBaseId=kb_id
            )["knowledgeBase"]
            status = response.get("status")
            if status in {None, "ACTIVE"}:
                return response
            if status in {"FAILED", "DELETE_UNSUCCESSFUL"}:
                raise RuntimeError(
                    f"Knowledge Base {kb_id} is not usable: status={status}"
                )
            time.sleep(2)
        raise TimeoutError(f"Knowledge Base {kb_id} did not become ACTIVE")

    @retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
    def create_structured_knowledge_base(self):
        try:
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name = self.kb_name,
                description = self.kb_description,
                roleArn = self.bedrock_kb_execution_role['Role']['Arn'],
                knowledgeBaseConfiguration = self.kbConfigParam,
                tags = WORKSHOP_TAGS_MAP,
            )
            kb = create_kb_response["knowledgeBase"]
            pp.pprint(kb)
        except self.bedrock_agent_client.exceptions.ConflictException:
            kb = self._get_knowledge_base_by_name()
            pp.pprint(kb)

        kb = self._wait_for_knowledge_base_active(kb)

        # create Data Sources
        print("Creating Data Sources aka query engine")
        try:
            create_ds_response = self.bedrock_agent_client.create_data_source(
            dataSourceConfiguration= {
                "type": "REDSHIFT_METADATA"
            },
            name=f"{self.kb_name}-ds",
            description="Query engine" ,
            knowledgeBaseId=kb['knowledgeBaseId']
        )
            
            ds = create_ds_response['dataSource']
            pp.pprint(ds)
        except self.bedrock_agent_client.exceptions.ConflictException:
            matching = [
                summary for summary in self._list_data_sources(kb["knowledgeBaseId"])
                if summary.get("name") == f"{self.kb_name}-ds"
            ]
            if not matching or not matching[0].get("dataSourceId"):
                raise RuntimeError(
                    f"Data source {self.kb_name!r}-ds was not visible after a conflict"
                )
            get_ds_response = self.bedrock_agent_client.get_data_source(
                dataSourceId=matching[0]["dataSourceId"],
                knowledgeBaseId=kb['knowledgeBaseId']
            )
            ds = get_ds_response["dataSource"]
            pp.pprint(ds)
       
        return kb, ds
    
    def start_ingestion_job(self):
       
        try:
            #  print(self.data_source)
            #  print(self.structured_knowledge_base['knowledgeBaseId'])
            start_job_response = self.bedrock_agent_client.start_ingestion_job(
                knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                dataSourceId=self.data_source["dataSourceId"]
            )
            job = start_job_response["ingestionJob"]
            print(f"job  started successfully\n")
            # pp.pprint(job)
            while job['status'] not in ["COMPLETE", "FAILED", "STOPPED"]:
                get_job_response = self.bedrock_agent_client.get_ingestion_job(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                    dataSourceId=self.data_source["dataSourceId"],
                    ingestionJobId=job["ingestionJobId"]
                )
                job = get_job_response["ingestionJob"]
            pp.pprint(job)
            if job["status"] != "COMPLETE":
                raise RuntimeError(
                    f"Ingestion job {job.get('ingestionJobId')} ended with "
                    f"status {job['status']}: {job.get('failureReasons', [])}"
                )
            return job

        except Exception as e:
            raise RuntimeError("Couldn't complete structured ingestion job") from e
            

    def get_knowledge_base_id(self):
        pp.pprint(self.knowledge_base["knowledgeBaseId"])
        return self.knowledge_base["knowledgeBaseId"]


    def delete_kb(self, delete_iam_roles_and_policies=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            kb_id = self.knowledge_base["knowledgeBaseId"]
            ds_id_list = self._list_data_sources(kb_id)

            for idx, ds in enumerate(ds_id_list):
                try:
                    self.bedrock_agent_client.delete_data_source(
                        dataSourceId=ds_id_list[idx]["dataSourceId"],
                        knowledgeBaseId=kb_id
                    )
                    print("======== Data source deleted =========")
                except self.bedrock_agent_client.exceptions.ResourceNotFoundException:
                    print(f"Data source {ds_id_list[idx]['dataSourceId']} not found")
                except Exception as e:
                    raise RuntimeError(
                        f"Error deleting data source "
                        f"{ds_id_list[idx]['dataSourceId']}"
                    ) from e
            
            # delete KB
            try:
                self.bedrock_agent_client.delete_knowledge_base(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                )
                print("======== Knowledge base deleted =========")

            except self.bedrock_agent_client.exceptions.ResourceNotFoundException:
                print("Knowledge base not found")
            except Exception as e:
                raise RuntimeError("Error deleting structured Knowledge Base") from e

            for _ in range(30):
                try:
                    self.bedrock_agent_client.get_knowledge_base(
                        knowledgeBaseId=kb_id
                    )
                except self.bedrock_agent_client.exceptions.ResourceNotFoundException:
                    break
                time.sleep(2)
            
            # delete role and policies
            if delete_iam_roles_and_policies:
                self.delete_iam_role_and_policies()

    def delete_iam_role_and_policies(self):
        policies_to_detach = []
        marker = None
        while True:
            request = {"RoleName": self.bedrock_kb_execution_role_name}
            if marker:
                request["Marker"] = marker
            response = self.iam_client.list_attached_role_policies(**request)
            policies_to_detach.extend(response.get("AttachedPolicies", []))
            if not response.get("IsTruncated"):
                break
            marker = response.get("Marker")
            if not marker:
                raise RuntimeError(
                    f"IAM returned a truncated policy list without a marker for "
                    f"{self.bedrock_kb_execution_role_name}"
                )

        
        for policy in policies_to_detach:
            policy_name = policy['PolicyName']
            policy_arn = policy['PolicyArn']

            try:
                self.iam_client.detach_role_policy(
                    RoleName=self.kb_execution_role_name,
                    PolicyArn=policy_arn
                )
                if policy_arn.startswith(
                    f"arn:aws:iam::{self.account_number}:policy/"
                ):
                    self.iam_client.delete_policy(PolicyArn=policy_arn)
            except self.iam_client.exceptions.NoSuchEntityException:
                print(f"Policy {policy_arn} not found")

        try:
            self.iam_client.delete_role(RoleName=self.kb_execution_role_name)
        except self.iam_client.exceptions.NoSuchEntityException:
            print(f"Role {self.kb_execution_role_name} not found")
        
        print("======== All IAM roles and policies deleted =========")
