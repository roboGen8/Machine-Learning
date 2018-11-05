import subprocess
import sys
from os import environ
env = environ.copy() 

env['CLASSPATH'] = "C:\Users\Draper\Documents\OMSCS\Machine Learning\OPT\ABAGAIL1.jar"
env['JYTHON_HOME'] = 'C:\jython2.7.0'
env['PATH'] = "%s;C:\\jython2.7.0;C:\\jython2.7.0\\bin" % env['PATH']

#N = int(sys.argv[1])

cmd = "jython co_hill.py"
print(cmd)
subprocess.call(cmd, env=env, shell=True)

cmd = "jython co_sa.py"
print(cmd)
subprocess.call(cmd, env=env, shell=True)

cmd = "jython co_ga.py"
print(cmd)
subprocess.call(cmd, env=env, shell=True)

cmd = "jython co_mimic.py"
print(cmd)
subprocess.call(cmd, env=env, shell=True)



