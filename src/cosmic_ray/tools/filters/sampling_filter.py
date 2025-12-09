"""A filter that applies stratified sampling to the work database.

This filter reduces the number of mutations to be executed by applying a
two-level sampling strategy:
1. It groups all pending mutations by their module (file).
2. Within each module, it samples up to a maximum number of mutations,
   stratifying the sample across the available mutation operators to
   ensure diversity.

Any mutation not selected as part of the sample is marked as SKIPPED in
the work database.
"""
import logging
import random
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict

from cosmic_ray.config import load_config
from cosmic_ray.tools.filters.filter_app import FilterApp
from cosmic_ray.work_db import WorkDB
from cosmic_ray.work_item import WorkItem, WorkResult, WorkerOutcome

log = logging.getLogger()


class SamplingFilter(FilterApp):
    """Implements the sampling-filter."""

    def description(self):
        return __doc__

    def add_args(self, parser: ArgumentParser):
        parser.add_argument("config", help="Path to the Cosmic Ray configuration file")

    def filter(self, work_db: WorkDB, args: Namespace):
        """
        Apply stratified sampling to the pending work items in the database.
        Items not selected in the sample will be marked as SKIPPED.
        """
        config = load_config(args.config)

        # Corrected configuration loading
        try:
            # Attempt to get the specific config for this filter
            filter_config = config.sub("filters", "sampling-filter")
        except KeyError:
            # If the section doesn't exist in the config file, use an empty dict
            filter_config = {}

        max_per_module = filter_config.get("max-per-module", 30)

        log.info(
            "Applying sampling filter with max-per-module=%d",
            max_per_module
        )

        pending_items = list(work_db.pending_work_items)

        # 1. Group all pending WorkItems by module, then by operator
        items_by_module = defaultdict(lambda: defaultdict(list))
        for item in pending_items:
            # Assuming one mutation per work item, which is standard
            mutation = item.mutations[0]
            items_by_module[mutation.module_path][mutation.operator_name].append(item)

        # 2. Determine which items to KEEP based on sampling logic
        items_to_keep = set()
        for module_path, operators_in_module in items_by_module.items():
            num_operators = len(operators_in_module)
            if num_operators == 0:
                continue

            # Calculate the ideal number of items to sample per operator for this module
            ideal_share = max(1, max_per_module // num_operators)

            module_samples = []
            for operator_name, items in operators_in_module.items():
                if len(items) <= ideal_share:
                    # If the group is smaller than the ideal share, keep all of them
                    module_samples.extend(items)
                else:
                    # Otherwise, take a random sample
                    module_samples.extend(random.sample(items, ideal_share))

            # If we're still over the per-file limit, trim the sample down
            if len(module_samples) > max_per_module:
                module_samples = random.sample(module_samples, max_per_module)

            # Add the job_ids of the items we decided to keep to a set for fast lookup
            for item in module_samples:
                items_to_keep.add(item.job_id)

        # 3. Determine which items to SKIP
        job_ids_to_skip = [
            item.job_id for item in pending_items if item.job_id not in items_to_keep
        ]

        log.info(
            "Discovered %d pending mutations. Sampled %d to keep, skipping %d.",
            len(pending_items),
            len(items_to_keep),
            len(job_ids_to_skip),
        )

        # 4. Mark all non-selected items as SKIPPED in the database
        if job_ids_to_skip:
            work_db.set_multiple_results(
                job_ids_to_skip,
                WorkResult(
                    output="Skipped by sampling-filter",
                    worker_outcome=WorkerOutcome.SKIPPED,
                ),
            )


def main(argv=None):
    """Run the sampling-filter with the specified command line arguments."""
    return SamplingFilter().main(argv)


if __name__ == "__main__":
    sys.exit(main())