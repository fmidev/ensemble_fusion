#!/bin/bash

#module load Python
# Originally named apta_global_2015_srf_extract.sh
# Changed by Roux for Chinese test cases


set -e
set -u
set -x

da=false
da=true

variablesMET=""
if $da; then
  variables="cnc_O3_gas cnc_NO2_gas cnc_SO2_gas cnc_PM2_5 cnc_PM10 cnc_CO_gas cnc_NO_gas"; dssuff=""
else
  variables="cnc_O3_gas*1.2/air_dens cnc_NO2_gas*1.2/air_dens cnc_SO2_gas*1.2/air_dens cnc_PM2_5 cnc_PM10 cnc_CO_gas*1.2/air_dens cnc_NO_gas*1.2/air_dens"; dssuff="-ntp"
  #variablesMET="BLH lai g_stomatal Kz_1m cnc_NOX_gas*1.2/air_dens cnc_OX_gas*1.2/air_dens"
  variablesMET="cnc_NOX_gas*1.2/air_dens cnc_OX_gas*1.2/air_dens"
fi

cd /lustre/tmp/silam/EVAL2015/VRA2017/scripts
dataset=$1

   path_to_ds=/lustre/tmp/silam/EVAL2015/VRA2017/output
   ctl_file=$path_to_ds/${dataset}_nc/0PM.nc.ctl
   [ -f $ctl_file ] || exit -1

  #extract to joint datasets
  outd=${dataset}${dssuff}
  mkdir -p ../TimeVars/${outd}
  for v in $variables $variablesMET; do

      #echo true extracting $v db $db dataset $dataset 
      dbvar=`echo $v |sed -e s/PM2_5/PM25/ -e s/cnc_// -e s/_gas//  -e 's/\*.*$//'`
      ncvar=`echo $v |sed -e 's/cnc_NOX_gas/\(cnc_NO_gas*1.53+cnc_NO2_gas\)/' -e 's/cnc_OX_gas/\(cnc_NO2_gas*1.04+cnc_O3_gas\)/'`

      outfile=../TimeVars/${outd}/${outd}_${dbvar}.nc
      [ -f $outfile ] && continue  #Skip if done
      echo "python extract_ts2nc.py $ctl_file $outfile \\\"$ncvar\\\""
  done | aprun -n1 -d 56 -j 2 -cc none xargs -r -t -P 20 -I XXX sh -c "XXX"


aprun -n 56 -d1 -j2 -cc none python MakeStats_nc.py $dataset${dssuff}

