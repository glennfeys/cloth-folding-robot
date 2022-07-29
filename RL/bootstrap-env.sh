#! /bin/bash
# This script will download the latest release version of Unity ML Agents and
# install any of the Python dependencies we may need during development that
# cannot be readily grabbed from PyPi.

MLAGENTS_RELEASE_URI="https://github.com/Unity-Technologies/ml-agents/archive/release_12.zip"
DIR="ml-agents-release_12"
ZIP="release_12.zip"

package_exist(){
    package=$1
    if pip freeze | grep $package; then
        echo "found"
    else
        echo ""
    fi
}

install() {
    package=$1
    if [ ! -d "$DIR" ]; then
        # Test if zip is present and unzip in this case, else download
        if [ ! -f "$ZIP" ]; then
            wget $MLAGENTS_RELEASE_URI
        fi
        unzip $ZIP
    fi
    pip install -e "./$DIR/$1"

}

echo "SELab3: Robot Folding via RL"
echo "Bootstrap Python Environment"
echo ""

source env/bin/activate

if [[ ! $(package_exist gym-unity) ]]; then
echo "Package gym-unity is missing."
install gym-unity
fi

if [[ ! $(package_exist ml-agents-envs) ]]; then
echo "Package ml-agents-envs is missing."
install ml-agents-envs
fi

pip install -r requirements.txt
pip freeze --exclude-editable > requirements.txt
