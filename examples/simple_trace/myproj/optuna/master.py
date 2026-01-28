from __future__ import annotations

from optuna_framework.adapters.master import MasterAdapter


class TraceMasterAdapter(MasterAdapter):
    def init(self, context):
        print('[MASTER] init', context)

    def execute(self, context):
        print('[MASTER] execute', context)

    def finish(self, context):
        print('[MASTER] finish', context)
