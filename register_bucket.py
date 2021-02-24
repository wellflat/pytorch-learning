#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upload s3 folder(s) as AllegroAi dataset

Requirements:
- allegroai installed -> pip install allegroai
- ~/trains.conf file with
    1. api.credentials.access_key and api.credentials.secret_key fields.
    2. S3 credentials: sdk.aws.s3 section with relevant credentials info, e.g. one of
    {key: "", secret: "", region: ""} of credentials {} section

How to use the script:
~/upload_data_for_annotations.py -d "<dataset name>" -v "<version name>" -b <bucket 1> ... <bucket N>
"""
from argparse import ArgumentParser, Namespace
from typing import List
from pathlib2 import Path
from allegroai import DatasetVersion, SingleFrame, Task
import boto3

APPROVED_SUFFIX = [".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"]

def get_metadata(s3_client, bucket, key):
    s3_object = s3_client.get_object(Bucket=bucket, Key=key)
    return s3_object.get("Metadata", {})


def assume_role(aws_account_number: str, role_name: str) -> boto3.Session:
    """
    Assumes the provided role in the target account and returns Session.
    Args:
        - aws_account_number: AWS Account Number
        - role_name: Role to assume in target account
    Returns:
        AssumeRole Session.
    """
    try:
        sts_client = boto3.client('sts')

        # Get the current partition
        partition = sts_client.get_caller_identity()['Arn'].split(":")[1]

        response = sts_client.assume_role(
            RoleArn=f'arn:{partition}:iam::{aws_account_number}:role/{role_name}',
            RoleSessionName=f'SessionFor{role_name}In{aws_account_number}'
        )
        #print(response)

        # Storing STS credentials
        session = boto3.Session(
            aws_access_key_id=response['Credentials']['AccessKeyId'],
            aws_secret_access_key=response['Credentials']['SecretAccessKey'],
            aws_session_token=response['Credentials']['SessionToken']
        )
    except Exception as e:
        raise ValueError(f'Error in AssumeRole process: {e}')
      
    print(f'Assumed session for {role_name} in {aws_account_number}.')

    return session


def update_frames_from_bucket(buckets: List[str], annotated: bool, session: boto3.Session) -> List[SingleFrame]:
    """
    :param buckets: List of buckets full names without the s3 prefix, e.g. 'allegro-examples/example-folder' for
    https://s3.console.aws.amazon.com/s3/buckets/allegro-examples/example-folder
    :return: List of new frames to be uploaded
    """
    s3_client = session.client('s3')
    frames_to_upload = []
    for bucket_name in buckets:
        bucket_root_dir, _, bucket_prefix = bucket_name.partition("/")
        objects = s3_client.list_objects_v2(Bucket=bucket_root_dir, Prefix=bucket_prefix)
        location = s3_client.get_bucket_location(Bucket=bucket_root_dir)['LocationConstraint']
        bucket_objects = objects.get("Contents", [])  # List of all the specific bucket object in the form of dicts
        for entry in bucket_objects:
            #print(entry)
            file_key = entry.get('Key')
            size = entry.get('Size')
            hash = entry.get('ETag')
            timestamp = entry.get('LastModified').timestamp()
            #source_path = f"s3://{bucket_root_dir}/{file_key}"
            #source_path = f"https://{bucket_root_dir}.s3-{location}.amazonaws.com/{file_key}"
            source_path = f"https://rest-term.com/tmp/{bucket_root_dir}/{file_key}"
            print(source_path)
            if file_key and not source_path.endswith("/") and Path(file_key).suffix in APPROVED_SUFFIX:
                source_metadata = get_metadata(s3_client=s3_client, bucket=bucket_root_dir, key=file_key)
                frame = SingleFrame(
                    source=source_path,
                    metadata=source_metadata,
                    size=size,
                    hash=hash,
                    timestamp=int(timestamp)
                )
                if annotated:
                    cls = Path(file_key).parts[-2]
                    frame.add_annotation(frame_class=[cls])
                frames_to_upload.append(frame)
    return frames_to_upload


def upload_data_to_platform(dataset_name: str, version_name: str, annotated: bool, frames: List[SingleFrame]):
    """
    :param dataset_name: The basic DataSet name.
    :param version_name: Version name if the dataset.
    :param buckets: List of full paths to the buckets contain the data.
    """
    Task.init(
        project_name="Data registration",
        task_name="Register buckets",
        task_type=Task.TaskTypes.data_processing
    )
    # Get the version we want to update
    try:
        version = DatasetVersion.get_version(dataset_name=dataset_name, version_name=version_name)
    except ValueError:
        print(f"Can not find version {version_name}, will create a new one in {dataset_name} dataset")
        version = DatasetVersion.create_version(dataset_name=dataset_name, version_name=version_name)

    # Take the data from the bucket and upload it
    #buckets_frames = update_frames_from_bucket(buckets=buckets, annotated=annotated)
    version.add_frames(frames)
    print(f'{len(frames)} images registration completed')


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset-name', help='The name of the dataset', required=True)
    parser.add_argument('-v', '--version-name', help='The name for the version', required=True)
    parser.add_argument('-b', '--buckets', nargs='*',
                        help='The full path to the root bucket contains the files, no need the s3 prefix',
                        required=True)
    parser.add_argument('--annotated', type=bool, default=False,
                        help='If True, assumes images are in subfolders. Subfolders considered as labels')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    session = assume_role('042830681561', 'S3-ReadOnly')
    frames = update_frames_from_bucket(args.buckets, args.annotated, session)
    #print(frames)
    upload_data_to_platform(
        dataset_name=args.dataset_name,
        version_name=args.version_name,
        annotated=args.annotated,
        frames=frames
    )
