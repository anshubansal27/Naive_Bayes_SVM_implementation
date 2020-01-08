if [ $1 == 1 ] 
then
	python3 naive_bayes.py $2 $3 $4
elif [ $1 == 2 -a $4 == 0 ]
then
	python3  svm2.py $2 $3 $5
elif [ $1 == 2 -a $4 == 1 ]
then
	python3 svm2b.py $2 $3 $5
else 
	echo "ques number does not exist"

fi
