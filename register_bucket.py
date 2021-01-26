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
from argparse import ArgumentParser
from typing import List
from pathlib2 import Path

from allegroai import DatasetVersion, SingleFrame, Task
from trains.storage.helper import StorageHelper

#from registration_config import APPROVED_SUFFIX

APPROVED_SUFFIX = [".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"]

def get_metadata(s3_client, bucket, key):
    s3_object = s3_client.get_object(Bucket=bucket, Key=key)
    return s3_object.get("Metadata", {})


def update_frames_from_bucket(buckets,  # type: List
                              annotated,  # type: bool
                              ):
    """
    :param buckets: List of buckets full names without the s3 prefix, e.g. 'allegro-examples/example-folder' for
    https://s3.console.aws.amazon.com/s3/buckets/allegro-examples/example-folder
    :return: List of new frames to be uploaded
    """
    frames_to_upload = []
    for bucket_name in buckets:
        helper = StorageHelper.get(f"s3://{bucket_name}")
        bucket_client = helper._container.resource.meta.client  # pylint: disable=protected-access
        bucket_root_dir, _, bucket_prefix = bucket_name.partition("/")
        objects = bucket_client.list_objects_v2(Bucket=bucket_root_dir, Prefix=bucket_prefix)
        bucket_objects = objects.get("Contents", [])  # List of all the specific bucket object in the form of dicts
        for entry in bucket_objects:
            file_key = entry.get('Key')
            source_path = f"{bucket_root_dir}/{file_key}"
            if file_key and not source_path.endswith("/") and Path(file_key).suffix in APPROVED_SUFFIX:
                source_metadata = get_metadata(s3_client=bucket_client, bucket=bucket_root_dir, key=file_key)
                frame = SingleFrame(source=f"s3://{source_path}", metadata=source_metadata)
                if annotated:
                    cls = Path(file_key).parts[-2]
                    frame.add_annotation(frame_class=[cls])
                frames_to_upload.append(frame)
    return frames_to_upload


def upload_data_to_platform(dataset_name,  # type: str
                            version_name,  # type: str
                            buckets,  # type: List[str]
                            annotated,  # type: bool
                            ):
    """
    :param dataset_name: The basic DataSet name.
    :param version_name: Version name if the dataset.
    :param buckets: List of full paths to the buckets contain the data.
    """
    Task.init(project_name="Data registration", task_name="Register buckets")
    # Get the version we want to update
    try:
        version = DatasetVersion.get_version(dataset_name=dataset_name, version_name=version_name)
    except ValueError:
        print(f"Can not find version {version_name}, will create a new one in {dataset_name} dataset")
        version = DatasetVersion.create_version(dataset_name=dataset_name, version_name=version_name)

    # Take the data from the bucket and upload it
    buckets_frames = update_frames_from_bucket(buckets=buckets, annotated=annotated)
    version.add_frames(buckets_frames)


def add_args(parser):
    parser.add_argument('-d', '--dataset-name', help='The name of the dataset', required=True)
    parser.add_argument('-v', '--version-name', help='The name for the version', required=True)
    parser.add_argument('-b', '--buckets', nargs='*',
                        help='The full path to the root bucket contains the files, no need the s3 prefix',
                        required=True)
    parser.add_argument('--annotated', type=bool, default=False,
                        help='If True, assumes images are in subfolders. Subfolders considered as labels')


def parse_arguments():
    parser = ArgumentParser()
    add_args(parser)
    return parser


def main():
    parser = parse_arguments()
    upload_data_to_platform(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
