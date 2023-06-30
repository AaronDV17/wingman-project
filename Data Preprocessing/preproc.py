def leo_preproc(df=data):
    """Returns processed df"""
    
    from sklearn.preprocessing import OneHotEncoder
    
    df.loc[df['certs_held'] == 'Y', 'certs_held'], df.loc[df['certs_held'] == 'N', 'certs_held'] = 1, 0 # Y -> 1, N -> 0
    
    df.loc[df['dprt_apt_id'] == 'NONE', 'dprt_apt_id'], df.loc[df['dprt_apt_id'] == 'PVT', 'dprt_apt_id'] = 0, 0 # None and PVT -> 0
    df.loc[df['dprt_apt_id'] !=0, 'dprt_apt_id'] = 1 # values != 0 -> 1
    
    df.loc[df['dest_apt_id'] == 'NONE', 'dest_apt_id'], df.loc[df['dest_apt_id'] == 'PVT', 'dest_apt_id'] = 0, 0 # None and PVT -> 0
    df.loc[df['dest_apt_id'] !=0, 'dest_apt_id'] = 1 # values != 0 -> 1
    
    df['flt_plan_filed'].replace('UNK', 'NONE', inplace=True)
    df['flt_plan_filed'].replace('VFIF', 'IFR', inplace=True)
    df['flt_plan_filed'].replace(['CFR', 'MFR'], 'VFR', inplace=True)
    df['flt_plan_filed'].replace(['VFIF', 'MVFR'], 'NONE',inplace=True)
    
    cat = list(df['flt_plan_filed'].unique())
    
    ohe = OneHotEncoder()
    ohe_df = pd.DataFrame(ohe.fit_transform(df[['flt_plan_filed']]).toarray())
    ohe_df.columns = cat
    ohe_df
    df.drop(columns='flt_plan_filed')
    df = pd.concat([df, ohe_df], axis=1)

    df['pc_profession'].replace('UNK', 'No', inplace=True)
    df['pc_profession'].replace(['Yes', 'No'], [1, 0], inplace=True)
    return df
    return X_preproc, y_enc
