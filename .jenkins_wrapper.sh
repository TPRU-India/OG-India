from_config(){

    export $1=$(cat .regression.txt | grep $1 | sed 's/\s+//g' | cut -d" " -f2);
}
from_config numpy_version
from_config install_taxcalc_version
from_config compare_ogindia_version
from_config compare_taxcalc_version


wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
rm -rf $WORKSPACE/miniconda
bash miniconda.sh -b -p $WORKSPACE/miniconda
export PATH="$WORKSPACE/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update conda -n root
conda env list | grep ogindia_env && conda env remove -n ogindia_env || echo Didnt have to remove env
conda install nomkl
conda create --force -n ogindia_env python=2.7 nomkl

source activate ogindia_env
conda install --force yaml llvmlite enum34 funcsigs singledispatch libgfortran libpng openblas numba pytz pytest six toolz dateutil cycler scipy numpy=$numpy_version pyparsing pandas=0.18.1 matplotlib nomkl
conda remove mkl mkl-service || echo didnt have to remove mkl mkl-service
conda install --no-deps -c ospc taxcalc=$install_taxcalc_version --force
if [ "$ogindiainstallmethod" = "conda" ];then
    conda install -c ospc ogindia=$ogindiaversion
fi
if [ "$ogindiainstallmethod" = "git" ];then
    python setup.py install
fi

conda install nomkl

cd regression
echo RUN REFORMS
conda env list
conda list
ls -lrth
stat puf.csv
head -n 1 puf.csv
md5sum puf.csv

python run_reforms.py $reform $ogindiabranch
