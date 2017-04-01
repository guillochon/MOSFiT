"""Definitions for the `Utility` class."""
import numpy as np

from mosfit.modules.utilities.utility import Utility


# Important: Only define one ``Module`` class per file.


class Operator(Utility):
    """Template class for photosphere Modules."""

    ops = {
        '+': np.add,
        '-': np.subtract,
        '*': np.multiply,
        '/': np.divide
    }

    def set_attributes(self, task):
        """Set key replacement dictionary."""
        super(Operator, self).set_attributes(task)
        self._operands = task.get('operands', [])
        if not self._operands:
            raise ValueError('`Operator` must have at least one operand.')
        self._result = task.get('result', 'result')
        self._op = self.ops.get(task.get('operator', '+'), np.add)

    def process(self, **kwargs):
        """Process module."""
        ops = [
            'dense_' + x if self._wants_dense and
            not x.startswith('dense_') else x for x in self._operands]
        result = kwargs[ops[0]]
        for op in ops[1:]:
            result = self._op(result, kwargs[op])
        return {self.dense_key(self._result): result}
