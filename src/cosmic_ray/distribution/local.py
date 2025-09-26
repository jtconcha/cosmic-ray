"""Cosmic Ray distributor that runs tests sequentially and locally.

Enabling the distributor
========================

To use the local distributor, set ``cosmic-ray.distributor.name = "local"`` in your Cosmic Ray configuration:

.. code-block:: toml

    [cosmic-ray.distributor]
    name = "local"
"""

import logging

from cosmic_ray.distribution.distributor import Distributor
from cosmic_ray.mutating import mutate_and_test, get_mutations_only

log = logging.getLogger(__name__)


class LocalDistributor(Distributor):
    "The local distributor."

    def __call__(self, pending_work, test_command, timeout, _distributor_config, on_task_complete):
        for work_item in pending_work:
            result = get_mutations_only(
                mutations=work_item.mutations,
            )
            on_task_complete(work_item.job_id, result)
