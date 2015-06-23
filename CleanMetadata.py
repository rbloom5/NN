import pandas as pd 

def slice_and_clean():

	full_DF = pd.read_csv('tnfa_response_expression.txt')
	full_metadata_DF = pd.read_csv('tnfa_response_metadata.txt')

	full_metadata_DF.columns = full_DF.columns

	#Drop patients you don't want
	drop_list = []

	for sample in full_metadata_DF:
		
		if sample.split('._.')[0] == 'GSE58795':
			#Remove the placebo patients from GSE58795
			if full_metadata_DF[sample].loc['characteristics_ch1.4'].split(': ')[1] == 'Placebo':
				drop_list.append(sample)
			#Remove the moderate responders from GSE58795
			if full_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'moderate':
				drop_list.append(sample)

		if sample.split('._.')[0] == 'GSE15258':
			#Remove the MEDIUM responders
			if full_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'MEDIUM':
				drop_list.append(sample)
			#Remove the NA responders
			if full_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'NA':
				drop_list.append(sample)

	subset_DF = full_DF.drop(set(drop_list), axis=1)
	subset_metadata_DF = full_metadata_DF.drop(set(drop_list), axis=1)

	#Make responder list

	response_list = []

	response = 'error'

	for sample in subset_metadata_DF:

		if sample.split('._.')[0] == 'GSE12051':
			if subset_metadata_DF[sample].loc['characteristics_ch1'].split(': ')[1] == 'nonresponder':
				response = 0
			if subset_metadata_DF[sample].loc['characteristics_ch1'].split(': ')[1] == 'responder':
				response = 1

		if sample.split('._.')[0] == 'GSE58795':
			if subset_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'nonresponder':
				response = 0
			if subset_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'responder':
				response = 1

		if sample.split('._.')[0] == 'GSE15258':
			if subset_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'NORESPONSE':
				response = 0
			if subset_metadata_DF[sample].loc['characteristics_ch1.5'].split(': ')[1] == 'RESPONSE':
				response = 1

		if sample.split('._.')[0] == 'GSE33377':
			if subset_metadata_DF[sample].loc['characteristics_ch1.2'].split(' ')[2] == 'non-responder':
				response = 0
			if subset_metadata_DF[sample].loc['characteristics_ch1.2'].split(' ')[2] == 'responder':
				response = 1

		if sample.split('._.')[0] == 'GSE20690':
			if subset_metadata_DF[sample].loc['title'].split('_')[2].split(' ')[0] == 'Residual':
				response = 0
			if subset_metadata_DF[sample].loc['title'].split('_')[2].split(' ')[0] == 'No':
				response = 1

		response_list.append(response)

	return subset_DF, response_list






