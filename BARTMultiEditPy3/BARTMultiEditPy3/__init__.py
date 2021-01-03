import jpype.imports
from pathlib import Path

base_path = Path(__file__).parent
relative_jar_path = '../../out/artifacts/BARTMultiEdit_jar/BARTMultiEdit.jar'
absolute_jar_path = (base_path/relative_jar_path).resolve();
jpype.startJVM(classpath=[str(absolute_jar_path)])

if jpype.isJVMStarted():
    print('JVM started: ', jpype.getJVMVersion())