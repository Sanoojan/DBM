#!/usr/bin/env python3
"""
Download a local copy of a Processed/* dataset from s3, according to a given config
"""

import argparse
import datetime
import fnmatch
import glob
import os
import subprocess
from pathlib import Path

import boto3
import botocore
import yaml


def get_disk_space(dir):
    """Return available local disk space in bytes."""
    if not os.path.exists(dir):
        print(f"{dir} does not exist, creating it.")
        os.makedirs(dir)
    statvfs = os.statvfs(dir)
    return statvfs.f_frsize * statvfs.f_bavail


def matches_filters(key, exclude_patterns, include_patterns, s3_prefix):
    return (
        not any(fnmatch.fnmatch(key, os.path.join(s3_prefix, pattern)) for pattern in exclude_patterns)
        and not any(fnmatch.fnmatch(key, pattern) for pattern in exclude_patterns)
        and (
            any(fnmatch.fnmatch(key, os.path.join(s3_prefix, pattern)) for pattern in include_patterns)
            or any(fnmatch.fnmatch(key, pattern) for pattern in include_patterns)
        )
    )


def compute_disk_space_required(objects, exclude_patterns, include_patterns, s3_prefix):
    total_size = 0
    for obj in objects:
        key = obj["Key"]
        size = obj["Size"]
        if matches_filters(key, exclude_patterns, include_patterns, s3_prefix):
            total_size += size
    return total_size


def list_s3_objects(bucket, prefix, s3_client):
    """List all objects in an S3 bucket with the given prefix."""
    objects = []
    continuation_token = None
    while True:
        list_args = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            list_args["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_args)
        objects.extend(response.get("Contents", []))

        # Check if there are more objects to retrieve
        if response.get("IsTruncated"):  # True if there are more objects
            continuation_token = response.get("NextContinuationToken")
        else:
            break
    return objects


def sync_s3_to_local(
    bucket,
    s3_prefix,
    local_path,
    exclude_patterns="",
    include_patterns="*",
    dryrun=True,
    debug=False,
    profile_name=None,
    use_boto3=False,
    use_credentials=False,
    cfg=None,
):
    """Sync files from S3 to local directory. Set profile_name="ec2" for ec2 usage."""
    if s3_prefix[-1] != "/":
        s3_prefix += "/"

    # Create session and client
    session_args = {}
    if profile_name is not None:
        session_args["profile_name"] = profile_name
    session = boto3.Session(**session_args)
    if use_credentials:
        s3_client = session.client("s3")
    else:
        s3_client = session.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))

    # Try to get list of objects
    try:
        objects = list_s3_objects(bucket, s3_prefix, s3_client=s3_client)
    except:
        print(f"Could not list s3 objects, check credentials?")
        return

    # Confirm download
    total_size = compute_disk_space_required(objects, exclude_patterns, include_patterns, s3_prefix)
    available_space = get_disk_space(local_path)
    user_question = f"This will download {total_size/2**30:.1f} GB to {local_path}. Available space: {available_space/2**30:.1f} GB.\n"
    user_question += "By downloading this dataset, you agree to adhere to its license - CC BY-NC-SA 4.0.\n"
    user_question += "You are free to share and adapt the dataset, but only for non-commercial purposes.\n"
    user_question += (
        "You must provide attribution and must share any derived material under the same license as the original.\n"
    )
    user_question += "The full license is available at https://creativecommons.org/licenses/by-nc-sa/4.0/ and will also be downloaded with the dataset.\n"
    user_question += "Do you agree and want to continue downloading? (y/n)"
    user_input = input(user_question).strip().lower()
    if user_input != "y":
        return

    # Loop through objects and download
    if use_boto3:
        for obj in objects:
            key = obj["Key"]

            # Check exclusion and inclusion filters - must match at least one inclusion pattern and no exclusion patterns
            if not matches_filters(key, exclude_patterns, include_patterns, s3_prefix):
                continue

            # Check if file already exists locally
            # https://stackoverflow.com/questions/72302266/is-it-possible-to-run-aws-s3-sync-with-boto3
            local_file_path = os.path.join(local_path, key.split(s3_prefix)[-1])
            meta_data = s3_client.head_object(Bucket=bucket, Key=key)
            if os.path.isfile(local_file_path):
                # Get modified times for comparison
                s3_last_modified = meta_data["LastModified"].replace(tzinfo=datetime.timezone.utc)
                local_last_modified = datetime.datetime.fromtimestamp(
                    os.path.getmtime(local_file_path), datetime.timezone.utc
                )
                local_last_modified = local_last_modified.replace(tzinfo=datetime.timezone.utc)

                # Get sizes for comparison
                s3_size_bytes = obj["Size"]
                local_size_bytes = Path(local_file_path).stat().st_size

                # Skip if same or newer local version and same size
                if local_last_modified > s3_last_modified and s3_size_bytes == local_size_bytes:
                    if debug:
                        print(
                            f"Skipping since newer local version and same file size: {bucket}/{key}\n  --> {local_file_path}"
                        )
                    continue

            # Check for existing directories
            if os.path.isdir(local_file_path):
                if debug:
                    print(f"Skipping since directory already exists: {bucket}/{key}\n  --> {local_file_path}")
                continue

            # Download the file
            if dryrun:
                print(f"Download: {bucket}/{key}\n  --> {local_file_path}")
            else:
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3_client.download_file(bucket, key, local_file_path)
                print(f"Downloaded {key} to {local_file_path}")
    else:
        cmd = ["aws", "s3", "sync"]
        cmd.append(f"s3://{bucket}/{s3_prefix}")
        cmd.append(local_path)
        if profile_name is not None:
            cmd.append("--profile")
            cmd.append(profile_name)
        if dryrun:
            cmd.append("--dryrun")
        if debug:
            cmd.append("--debug")
        if not use_credentials:
            cmd.append("--no-sign-request")
        cmd.append("--exclude")
        cmd.append("*")
        for pattern in include_patterns:
            cmd.append("--include")
            cmd.append(pattern)
        for pattern in exclude_patterns:
            cmd.append("--exclude")
            cmd.append(pattern)
        subprocess.call(cmd)

    # Display total transfer at the end
    if dryrun:
        print(f"Required: {total_size/2**30:.1f} GB, Available: {available_space/2**30:.1f} GB.")
    else:
        available_space = get_disk_space(local_path)
        print(f"Copied volume: {total_size/2**30:.1f} GB, Space remaining: {available_space/2**30:.1f} GB.")


def read_yaml(file_path):
    assert os.path.exists(file_path), print(f"{file_path} does not exist.")
    _, ext = os.path.splitext(file_path)
    assert ext.lower() in (".yaml", ".yml"), print(f"{file_path} is not a yaml or yml file.")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    """
    Download an s3 dataset based on information in a config file, e.g.
      python data/download_processed_dataset.py \
        --cfg [default: configs/public-dataset.yaml] \
        --output_dir [default: ~/data/IDD] \
        --subject_ids [default: all subjects]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="data/configs/public-dataset.yaml",
        help="path to dataset config file",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="sample",
        help="version of the dataset to download (in cfg file, default: sample)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output directory (parent dir of Processed)",
    )
    parser.add_argument(
        "--profile_name",
        type=str,
        default=None,
        help="AWS profile name. Defaults to None, which does not use a profile.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print downloads but do not download.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional debug information.",
    )
    parser.add_argument(
        "--use_boto3",
        action="store_true",
        help="Use boto3 to actually download the files. By default use the AWS CLI.",
    )
    parser.add_argument(
        "--use_credentials",
        action="store_true",
        help="Use AWS credentials. By default, does not sign the request (good for public data).",
    )
    args = parser.parse_args()
    print(f'Using the config file "{args.cfg}" and the version "{args.version}".')

    # read config file
    cfg = read_yaml(args.cfg)

    # check for blank include_patterns
    if "include_patterns" not in cfg[args.version]:
        cfg[args.version]["include_patterns"] = ["*"]

    # check for blank exclude_patterns
    if "exclude_patterns" not in cfg[args.version]:
        cfg[args.version]["exclude_patterns"] = [""]

    # check for no configured dataset path
    if "s3_dataset_path" not in cfg:
        cfg["s3_dataset_path"] = ""
    if "local_dataset_path" not in cfg[args.version]:
        cfg[args.version]["local_dataset_path"] = ""

    # sync data from s3 to local disk
    sync_s3_to_local(
        bucket=cfg["s3_bucket_name"],
        s3_prefix=os.path.join(cfg["s3_project_folder"], cfg["s3_dataset_path"]),
        local_path=os.path.join(args.output_dir, cfg[args.version]["local_dataset_path"]),
        exclude_patterns=cfg[args.version]["exclude_patterns"],
        include_patterns=cfg[args.version]["include_patterns"],
        dryrun=args.dryrun,
        debug=args.debug,
        profile_name=args.profile_name,
        use_boto3=args.use_boto3,
        use_credentials=args.use_credentials,
        cfg=cfg,
    )
