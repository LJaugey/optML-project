#C=adgd_0.02_2.0_

C=sgdm_0.02_1_
#C=adgd_0.02_1_
# C=adgd_0.1_1.0_

for i in lenet5_seed_*/
do 
	cd $i  
	for j in ${C}*.npy
	do
		echo ${j#${C}}
		echo $j
		mv $j ${j#${C}} 
	done
	cd ..
done
