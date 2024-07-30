The datasets were adapted from P5 (https://github.com/jeykigung/P5/tree/main/preprocess), but they are not well documented. In this work, we removed redundant data entries and wrote some notes so as to improve the readability.

explanation.json
{
	'train': train,
	'val': val,
	'test': test,
}
train/val/test is a list of dictionaries, each of which is formatted as follows.
{
	'user': user, # int
	'item': item, # int
	'explanation': explanation, # an explanation sentence from the user's review
	'feature': feature, # a feature in the explanation sentence
}