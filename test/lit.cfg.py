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

# Add LLVM tools (FileCheck) to PATH
llvm_tools_dir = os.path.join(os.environ.get('CONDA_PREFIX', '/home/qianxu/miniconda3'),
                               'libexec', 'llvm')
if os.path.isdir(llvm_tools_dir):
    config.environment['PATH'] = llvm_tools_dir + os.pathsep + config.environment.get(
        'PATH', os.environ.get('PATH', ''))
