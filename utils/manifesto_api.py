if __name__ == '__main__':
    
    import json
    import os
    import pandas as pd
    import urllib.request
    
    API_KEY = os.environ['MANIFESTO_API']
    
    def query_manifesto(query, *args, api_key=API_KEY):
        try:
            root = "https://manifesto-project.wzb.eu/api/v1/"

            hdr ={
            # specifying requested encoding
            'Cache-Control': 'no-cache',
            'charset': 'UTF-8'
            }
            
            query_url = root + query + "?api_key=" + api_key
            
            if args: # concatenate variable arguments to url
                query_url += '&' + '&'.join(args)
            
            req = urllib.request.Request(query_url, headers=hdr)

            req.get_method = lambda: 'GET'
            response = urllib.request.urlopen(req)
            
            print(response.status) # print response
            
            return response
            
        except Exception as e:
            print(e)
    
           
    def build_manifesto_df(manif_dic):
        
        manifesto_id = manif_dic['key']
        manif_df = pd.DataFrame.from_records(manif_dic['items'])
        manif_df["manifesto_id"] = manifesto_id
        manif_df.reset_index(inplace=True)
        
        return manif_df

    
    resp = query_manifesto('get_core', 'key=MPDS2022a', 'kind=dta', 'raw=true')

    data = pd.read_stata(resp, convert_categoricals=False)

    codes = ['_'.join([str(p), str(d)]) for p, d in zip(data.party, data.date)]


    meta_data = [json.loads(query_manifesto('metadata', f'keys[]={c}','version=2022-1').read()) for c in codes]

    meta_df = pd.DataFrame.from_records([m['items'][0] for m in meta_data])
    meta_df.election_date = meta_df.election_date.astype('int64')

    data = data.merge(meta_df, how='inner', left_on=['date', 'party'], right_on=['election_date', 'party_id'])
    data_eng = data[(data.language == 'english') & (data.annotations == True)]

    text_annos = [json.loads(query_manifesto('texts_and_annotations', f'keys[]={c}', 'version=2022-1').read())['items'][0] for c in data_eng.manifesto_id]


    def build_manifesto_df(manif_dic):
        manifesto_id = manif_dic['key']
        manif_df = pd.DataFrame.from_records(manif_dic['items'])
        manif_df["manifesto_id"] = manifesto_id
        manif_df.reset_index(inplace=True)
        return manif_df


    data_final = data_eng.merge(pd.concat(list(map(build_manifesto_df, text_annos)), 
                                        axis=0, ignore_index=True), on='manifesto_id', how='inner')


    data_final.to_csv('manifestos.csv')
    
else:
    
    import json
    import os
    import pandas as pd
    import urllib.request
    API_KEY = os.environ['MANIFESTO_API']
    
    def query_manifesto(query, *args, api_key=API_KEY):
        try:
            root = "https://manifesto-project.wzb.eu/api/v1/"

            hdr ={
            # specifying requested encoding
            'Cache-Control': 'no-cache',
            'charset': 'UTF-8'
            }
            
            query_url = root + query + "?api_key=" + api_key
            
            if args: # concatenate variable arguments to url
                query_url += '&' + '&'.join(args)
            
            req = urllib.request.Request(query_url, headers=hdr)

            req.get_method = lambda: 'GET'
            response = urllib.request.urlopen(req)
            
            print(response.status) # print response
            
            return response
            
        except Exception as e:
            print(e)
    
           
    def build_manifesto_df(manif_dic):
        
        manifesto_id = manif_dic['key']
        manif_df = pd.DataFrame.from_records(manif_dic['items'])
        manif_df["manifesto_id"] = manifesto_id
        manif_df.reset_index(inplace=True)
        
        return manif_df
