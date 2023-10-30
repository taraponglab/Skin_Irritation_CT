def process_df(df, smiles_col, ic50_col, ic50_units, units):
    df = df[[smiles_col, ic50_col, ic50_units]] #select only the smiles and IC50 columns
    df = df.dropna() #drop missing values
    df = df[df[ic50_units] == units] #select only the rows with the specified units
    df = df.drop(ic50_units, axis=1) #drop IC50 units column
    df.index.name = "LigandID" #rename index
    return df

def check_missing_smiles(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import PandasTools
    #check missing smiles
    PandasTools.AddMoleculeColumnToFrame(df,smiles_col,'Molecule',includeFingerprints=True)
    P = PandasTools.FrameToGridImage(df,column= 'Molecule', molsPerRow=10,subImgSize=(300,300),legendsCol="LigandID", maxMols=1000)
    #print missing smiles
    print(df[df['Molecule'].isna()].index)
    #drop column Molecule
    df = df.drop('Molecule', axis=1)
    return df
def nanomolarconversion(df, ic50_col):
    import numpy as np
    #convert IC50 to float
    df[ic50_col] = df[ic50_col].astype(float)
    df[ic50_col] = df[ic50_col].div(1000000000) #convert nM to M
    return df 
def calculate_pic50(df, ic50_col):
    import numpy as np
    #change nM to M and calculate log
    df['pIC50'] = np.log10(df[ic50_col]) #take log of IC50
    df['pIC50'] = df['pIC50'].multiply(-1) #multiply by -1
    #round pIC50 to 2 decimal places
    df['pIC50'] = df['pIC50'].round(2)
    #drop IC50 column
    df = df.drop(ic50_col, axis=1)
    return df

def canonical_smiles(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #generate canonical smiles
    df['canonical_smiles'] = df[smiles_col].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True))
    return df

def has_carbon_atoms(smiles):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #check if molecule has carbon atoms
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        return len(carbon_atoms) > 0
    return False

def remove_inorganic(df, smiles_col):
    df['has_carbon'] = df[smiles_col].apply(has_carbon_atoms)
    df = df[df['has_carbon'] == True]
    df = df.drop('has_carbon', axis=1)
    return df

def remove_mixtures(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #check if molecule is a mixture using '.' as separator
    df['is_mixture'] = df[smiles_col].apply(lambda x: '.' in x)
    df = df[df['is_mixture'] == False]
    df = df.drop('is_mixture', axis=1)
    return df

def process_duplicates(df, smiles_col, pic50_col, threshold=0.2):
    df = df.reset_index(level=0) #reset index
    #first get the duplicate smiles
    duplicates = df[df.duplicated(subset=smiles_col, keep=False)].sort_values(smiles_col)
    #create array for storing index of rows to remove and rows to average
    to_remove = []
    to_average = []
    #create a loop to check the difference in pIC50 values
    for _, group in duplicates.groupby(smiles_col):
        pic50_diff = abs(group[pic50_col].max() - group[pic50_col].min())
        #if the difference is less than threshold, add the index to to_average
        if round(pic50_diff,2) <= threshold: #we need to round the difference to 2 decimal places otherwise it will not work
            to_average.append(group.index)
        else:
            to_remove.extend(group.index) #if the difference is greater than threshold, add the index to to_remove
            
    # Drop rows with pIC50 differences greater than threshold
    df = df.drop(index=to_remove)
    
    # Average pIC50 values and retain one entry for duplicates with pIC50 differences less than or equal to threshold
    for indices in to_average:
        avg_pic50 = df.loc[indices, pic50_col].mean()
        df = df.drop(index=indices[1:]) #Remove all the rows in the DataFrame df with indices present in to_average, except for the first index. 
        #indices[1:] selects all indices except the first one.
        df.loc[indices[0], pic50_col] = avg_pic50 #set the pIC50 value to the average value to the average value for the first index in to_average

    df = df.set_index('LigandID') #set index back to the original index
    return df

def remove_missingdata(df):
    #drop missing data
    df = df.dropna()
    return df

def save_duplicate_smiles(df1, df2, smiles_col):
    intersect_indice = df1[smiles_col].isin(df2[smiles_col])
    #save the intersection
    dup_df1 = df1.loc[intersect_indice]
    return dup_df1

def morded_cal(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    from rdkit import Chem
    from mordred import Calculator, descriptors
    #calculate descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col]]
    des = calc.pandas(mols)
    des = des.set_index(df.index)
    return des

def remove_constant_string_des(df):
    #delete string value
    df = df.select_dtypes(exclude=['object'])
    #delete constant value
    for column in df.columns:
        if df[column].nunique() == 1:  # This checks if the column has only one unique value
            df = df.drop(column, axis=1)  # This drops the column from the DataFrame
    return df

def get_name(ligandid):
    import pandas as pd
    import requests
    import xml.etree.ElementTree as ET
    base_url = 'https://www.ebi.ac.uk/chembl/api/data/molecule'
    response = requests.get(f"{base_url}/{ligandid}", verify=False)
    if response.status_code == 200:
        xml_data = response.text
        root = ET.fromstring(xml_data)
        name_element = root.find('pref_name')
        if name_element is not None:
            return name_element.text
    return None

#get DOI
def get_doi(document_id):
    import pandas as pd
    import requests
    import xml.etree.ElementTree as ET
    base_url = 'https://www.ebi.ac.uk/chembl/api/data/document'
    response = requests.get(f"{base_url}/{document_id}", verify=False)
    if response.status_code == 200:
        xml_data = response.text
        root = ET.fromstring(xml_data)
        doi_element = root.find('doi')
        if doi_element is not None:
            return doi_element.text
    return None

def lipinski_filter(df, canonical_smiles):
    #import RDKit
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    #canonical smiles to molecule
    df['Molecule'] = df[canonical_smiles].apply(lambda x: Chem.MolFromSmiles(x))
    #calculate descriptors
    df['MW'] = df['Molecule'].apply(Chem.Descriptors.MolWt)
    df['LogP'] = df['Molecule'].apply(Chem.Descriptors.MolLogP)
    df['HBA'] = df['Molecule'].apply(Chem.Descriptors.NumHAcceptors)
    df['HBD'] = df['Molecule'].apply(Chem.Descriptors.NumHDonors)
    #drop molecule column
    df = df.drop('Molecule', axis=1)
    #filter
    df = df[(df['MW'] <= 500) & (df['LogP'] <= 5) & (df['HBA'] <= 10) & (df['HBD'] <= 5)]
    return df

def lipinski_calculation_nofilter(df, canonical_smiles):
    #import RDKit
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    #canonical smiles to molecule
    df['Molecule'] = df[canonical_smiles].apply(lambda x: Chem.MolFromSmiles(x))
    #calculate descriptors
    df['MW'] = df['Molecule'].apply(Chem.Descriptors.MolWt)
    df['LogP'] = df['Molecule'].apply(Chem.Descriptors.MolLogP)
    df['HBA'] = df['Molecule'].apply(Chem.Descriptors.NumHAcceptors)
    df['HBD'] = df['Molecule'].apply(Chem.Descriptors.NumHDonors)
    #drop molecule column
    df = df.drop('Molecule', axis=1)
    #filter
    df = df[(df['MW'] <= 700)]
    df = df[(df['LogP'] <= 8)]
    return df