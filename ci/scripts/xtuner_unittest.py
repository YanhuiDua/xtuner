#!/usr/bin/env python3
"""
Yidian集群任务提交脚本
基于test_clusterx_sft.py的成功经验，专门用于yidian集群
保持与run.py相同的接口，但包含yidian特定的配置
"""
import sys
import time
import os
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clusterx import CLUSTER, CLUSTER_MAPPING
from clusterx.launcher.base import JobStatus

JOB_INFO_RETRY_TIMES = 6
JOB_INFO_RETRY_INTERVAL_S = 10


def _is_too_many_requests(exc: BaseException) -> bool:
    if getattr(exc, "status", None) == 429:
        return True
    text = f"{getattr(exc, 'body', '')} {exc}"
    return "TooManyRequests" in text or "scheduler list capacity is full" in text


def _retry_wait_s(exc: BaseException | None) -> int:
    wait = JOB_INFO_RETRY_INTERVAL_S
    headers = getattr(exc, "headers", None) if exc is not None else None
    retry_after = None
    if headers is not None:
        retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            wait = max(wait, int(retry_after))
        except (TypeError, ValueError):
            pass
    return wait


def parse_args():
    """解析命令行参数，与run.py保持一致"""
    parser = argparse.ArgumentParser(description="Yidian Cluster Submission")
    parser.add_argument("image", type=str)
    parser.add_argument("env", type=str)
    parser.add_argument("cmd", type=str)
    parser.add_argument("--nodes", type=int, default=1)
    return parser.parse_args()


def print_job_log(cluster, job_id: str) -> None:
    """Fetch logs immediately while replicas still exist after Failed."""
    try:
        log = cluster.get_log(job_id)
        print("=== 任务失败日志 ===")
        print(log)
    except Exception as e:
        print(f"获取日志失败: {e}")


def export_rjob_id(rjob_id: str) -> None:
    """Expose the suffixed rjob name to later GitHub Actions steps."""
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if not gh_out:
        return
    with open(gh_out, "a", encoding="utf-8") as f:
        f.write(f"rjob_id={rjob_id}\n")


def _is_valid_job_info(job_info) -> bool:
    return job_info is not None and getattr(job_info, "status", None) is not None


def get_job_info_with_retry(cluster, job_id: str):
    """Retry get_job_info until a job with status is returned.

    Transient 429 / scheduler-capacity errors keep the outer poll alive instead of
    failing a still-queuing or running job.
    """
    last_error = None
    rate_limited = False
    for attempt in range(1, JOB_INFO_RETRY_TIMES + 1):
        job_info = None
        try:
            job_info = cluster.get_job_info(job_id)
        except Exception as e:
            last_error = e
            rate_limited = _is_too_many_requests(e)
            print(
                f"get_job_info({job_id}) failed on attempt "
                f"{attempt}/{JOB_INFO_RETRY_TIMES}: {e}"
            )
        if _is_valid_job_info(job_info):
            return job_info
        if job_info is not None:
            print(
                f"get_job_info({job_id}) returned invalid job on attempt "
                f"{attempt}/{JOB_INFO_RETRY_TIMES}: {job_info!r}"
            )
        if attempt < JOB_INFO_RETRY_TIMES:
            wait_s = _retry_wait_s(last_error)
            print(f"Retry get_job_info in {wait_s}s")
            time.sleep(wait_s)

    detail = f": {last_error}" if last_error else ""
    message = (
        f"Job {job_id} info unavailable after {JOB_INFO_RETRY_TIMES} retries{detail}"
    )
    if rate_limited:
        print(message)
        return None
    raise RuntimeError(message)


def main():
    """主函数"""
    args = parse_args()

    cluster_spec = CLUSTER_MAPPING[CLUSTER]
    cluster_cls = cluster_spec["type"]
    params_cls = cluster_spec["params"]

    current_dir = os.getcwd()
    cmd = f"cd {current_dir}; {args.env}; {args.cmd}"

    assert (
        params_cls is not None and cluster_cls is not None
    ), f"Cluster {CLUSTER} is not available in current ci machine!"

    commit_id = os.environ.get("GITHUB_SHA", "test")[0:8]
    commit_branch = os.environ.get("GITHUB_HEAD_REF", "test")
    job_id = (
        os.environ.get("GITHUB_RUN_ID", "0")
        + "-"
        + os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    )
    commit_branch = commit_branch.replace("/", "-").replace("\\", "-").replace("_", "-")

    params = params_cls(
        job_name=f"xtuner-ci-{job_id}-{commit_id}",
        image=args.image,
        cmd=cmd,
        gpus_per_task=8,
        cpus_per_task=120,
        memory_per_task="800",
        num_nodes=args.nodes,
        priority=5,
        no_env=True,
    )

    print(f"创建集群任务: {params.job_name}")
    print(f"节点数: {params.num_nodes}")
    print(f"每节点GPU数: {params.gpus_per_task}")
    print(f"每节点CPU数: {params.cpus_per_task}")
    print(f"每节点内存: {params.memory_per_task}G")

    cluster = cluster_cls()
    job_schema = cluster.run(params)
    print(f"任务已提交: {job_schema.job_id}")
    export_rjob_id(job_schema.job_id)

    while True:
        time.sleep(10)

        job_info = get_job_info_with_retry(cluster, job_schema.job_id)
        if job_info is None:
            print(f"Job {job_schema.job_id} status temporarily unavailable, waiting...")
            continue
        status = job_info.status

        if status == JobStatus.QUEUING:
            print(f"Job {job_schema.job_id} is queuing...")
        elif status == JobStatus.RUNNING:
            print(f"Job {job_schema.job_id} is running...")
        elif status == JobStatus.STOPPED:
            print(f"Job {job_schema.job_id} is stopped...")
            print_job_log(cluster, job_schema.job_id)
            raise RuntimeError(f"Job {job_schema.job_id} failed with status {status}")
        elif status == JobStatus.FAILED:
            print(f"Job {job_schema.job_id} failed!")
            print_job_log(cluster, job_schema.job_id)
            raise RuntimeError(f"Job {job_schema.job_id} failed with status {status}")
        elif status == JobStatus.SUCCEEDED:
            print(f"Job {job_schema.job_id} succeeded!")
            break
        else:
            print(f"Found unrecognized status {status}, waiting...")


if __name__ == "__main__":
    main()
