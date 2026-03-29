import os
import lit.formats

config.name = "NPU"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, '..', 'build')

npu_opt = os.path.join(config.test_exec_root, 'tools', 'npu-opt', 'npu-opt')
config.substitutions.append(('%npu-opt', npu_opt))
config.substitutions.append(('%mlir-opt', 'mlir-opt'))
