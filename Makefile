#############################
###### SEL 3 Makefile ######
###########################

# Note: Set the path to your Unity executable in an environment variable UNITY_PATH
# e.g. macOS: "export UNITY_PATH=/Applications/Unity/Hub/Editor/2020.2.6f1/Unity.app/Contents/MacOS/Unity"
# e.g. Linux: "export UNITY_PATH=/home/___USERNAME___/Unity/Hub/Editor/2020.2.3f1/Editor/Unity"

PROJECT_PATH = Unity_Simulation/
BUILD_PATH = Build/
RL_PATH = RL/
LICENSE_PATH = Unity_Simulation/Unity_v2020.x.ulf

# Only Linux and macOS are supported in this Makefile
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	BUILD_TYPE = -buildLinux64Player
	EXECUTABLE_NAME = SEL3
endif
ifeq ($(UNAME_S),Darwin)
	BUILD_TYPE = -buildOSXUniversalPlayer
	EXECUTABLE_NAME = SEL3.app
endif

all: executable

# See: https://docs.unity3d.com/Manual/CommandLineArguments.html
executable:
	${UNITY_PATH} -projectPath ${PROJECT_PATH} -batchmode -quit -nographics -manualLicenseFile ${LICENSE_PATH}
	${UNITY_PATH} -projectPath ${PROJECT_PATH} ${BUILD_TYPE} ${BUILD_PATH}${EXECUTABLE_NAME} -batchmode -quit -nographics

clean:
	rm -r "${PROJECT_PATH}${BUILD_PATH}"

rl-bootstrap:
	cd ${RL_PATH}; chmod +x bootstrap-env.sh; ./bootstrap-env.sh
