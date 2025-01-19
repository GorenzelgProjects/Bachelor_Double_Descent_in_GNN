import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# List to hold dataframes
folder_path1 = "./Run1"
folder_path2 = "./Run2"
folder_path3 = "./Run3"


def read_files(folder_path,run_number, save = False):
    dfs = []

    # Iterate through the folder and read each CSV file
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            print(f"Reading file: {filename}")
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df['Train Error'] = 1 - df['Train Accuracy']/100
            df['Test Error'] = 1 - df['Test Accuracy']/100


            dfs.append(df)
    #print(dfs)
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    #save the concatenated dataframe to a csv file
    if save is True: 
        df.to_csv('concatenated_{}.csv'.format(run_number))

    return df

def create_Mlist(df):
    Mlist = {}
    ks = []
    

    
    dataset_fractions = df['Dataset Fraction'].unique()
    for i,fraction in enumerate(dataset_fractions):
        df = df.copy() 
        print('df:',df) 
        df = df[df['Dataset Fraction'] == fraction].copy()
        df['Model Size'] = df['Model Size'].astype(int)
        #last_epoch_df = df.loc[df.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
        if i == 1:
            df_test = df.copy()
            #df_test = df_test.loc[df_test.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
            df_test['Train Error'] = 1 - df_test['Train Accuracy']/100
            df_test['Test Error'] = 1 - df_test['Test Accuracy']/100
            df = df_test['Model Size'].copy()
            extract = df.to_list()
            ks.append(extract)
            df = df_test['Test Error'].copy()
            extract = df.to_list()
            Mlist['Test Error'] = extract
            df = df_test['Test Loss'].copy()
            extract = df.to_list()
            Mlist['Test Loss'] = extract
            df = df_test['Train Error'].copy()
            extract = df.to_list()
            Mlist['Train Error'] = extract
            df = df_test['Train Loss'].copy()
            extract = df.to_list()
            Mlist['Train Loss'] = extract
            
    return Mlist,ks

def epochs1(df, plot=False):
    #Take the last 10 epochs and calculate the mean for a given df column
    #
    # Convert 'Model Size' column to integers
    df['Model Size'] = df['Model Size'].astype(int)

    # Filter the dataframe to get the last epoch for each data fraction and model size
    last_epoch_df = df.loc[df.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
    #print(last_epoch_df)

    # get unique dataset fractions
    dataset_fractions = last_epoch_df['Dataset Fraction'].unique()

    # create 3 dataframe for each dataset fraction
    df_1 = last_epoch_df[last_epoch_df['Dataset Fraction'] == dataset_fractions[0]].copy()
    df_2 = last_epoch_df[last_epoch_df['Dataset Fraction'] == dataset_fractions[1]].copy()
    df_3 = last_epoch_df[last_epoch_df['Dataset Fraction'] == dataset_fractions[2]].copy()
    
    

    if plot is True:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

        # Plot for df_1
        axes[0].plot(df_1['Model Size'], df_1['Train Loss'], marker='o')
        axes[0].set_title(f'Train Loss vs Model Size for Dataset Fraction {dataset_fractions[0]}')
        axes[0].set_xlabel('Model Size')
        axes[0].set_ylabel('Train Loss')

        # Plot for df_2
        axes[1].plot(df_2['Model Size'], df_2['Train Loss'], marker='o')
        axes[1].set_title(f'Train Loss vs Model Size for Dataset Fraction {dataset_fractions[1]}')
        axes[1].set_xlabel('Model Size')
        axes[1].set_ylabel('Train Loss')

        # Plot for df_3
        axes[2].plot(df_3['Model Size'], df_3['Train Loss'], marker='o')
        axes[2].set_title(f'Train Loss vs Model Size for Dataset Fraction {dataset_fractions[2]}')
        axes[2].set_xlabel('Model Size')
        axes[2].set_ylabel('Train Loss')

        plt.tight_layout()
        plt.show()

    
    #df = df.copy()
    #df[column] = df[column].tail(10).mean()
    return df

def plot_test(dfs, plot=False,save_values=False):
    #df['Model Size'] = df['Model Size'].astype(int)
    Mlist = {}
    ks = []
    # Filter the dataframe to get the last epoch for each data fraction and model size
    #last_epoch_df = df.loc[df.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
    #print(last_epoch_df)

    # get unique dataset fractions
    #dataset_fractions = last_epoch_df['Dataset Fraction'].unique()
    dataset_fractions = dfs[0]['Dataset Fraction'].unique()
    # create a dictionary to hold dataframes for each dataset fraction
    dfs_dict = {}
    for i,fraction in enumerate(dataset_fractions):
        df = dfs[i].copy() 
        df['Model Size'] = df['Model Size'].astype(int)
        last_epoch_df = df.loc[df.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
        dfs_dict[i] = last_epoch_df[last_epoch_df['Dataset Fraction'] == fraction].copy()
        if i == 1:
            df_test = df.copy()
            df_test = df_test.loc[df_test.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
            df_test['Train Error'] = 1 - df_test['Train Accuracy']/100
            df_test['Test Error'] = 1 - df_test['Test Accuracy']/100
            df = df_test['Model Size'].copy()
            extract = df.to_list()
            ks.append(extract)
            df = df_test['Test Error'].copy()
            extract = df.to_list()
            Mlist['Test Error'] = extract
            df = df_test['Test Loss'].copy()
            extract = df.to_list()
            #extract = np.array(extract)
            Mlist['Test Loss'] = extract
            df = df_test['Train Error'].copy()
            extract = df.to_list()
            Mlist['Train Error'] = extract
            df = df_test['Train Loss'].copy()
            extract = df.to_list()
            Mlist['Train Loss'] = extract
            
            
            
        #dataset_fractions = last_epoch_df['Dataset Fraction'].unique()
    #dataset_fractions = last_epoch_df['Dataset Fraction'].unique()
    #df_mean = last_epoch_df.copy()
    
    for df in dfs:
        df_mean = df.copy()
        df_mean['Train Loss'] += df['Train Loss']
        df_mean['Train Accuracy'] += df['Train Accuracy']
        df_mean['Test Loss'] += df['Test Loss']
        df_mean['Test Accuracy'] += df['Test Accuracy']
    
    df_mean['Train Loss'] /= len(dfs)
    df_mean['Train Accuracy'] /= len(dfs)
    df_mean['Test Loss'] /= len(dfs)
    df_mean['Test Accuracy'] /= len(dfs)
    df_mean['Train Error'] = 1 - df_mean['Train Accuracy']/100
    df_mean['Test Error'] = 1 - df_mean['Test Accuracy']/100
    # Model Size and Dataset Fraction remain the same
    if plot is True:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        last_epoch_df = df_mean.loc[df_mean.groupby(['Model Size', 'Dataset Fraction'])['Epoch'].idxmax()]
        for i, dataset_fraction in enumerate(dataset_fractions):
            df_temp = last_epoch_df[last_epoch_df['Dataset Fraction'] == dataset_fraction]
            print(df_temp['Model Size'])
            axes[i].plot(df_temp['Model Size'], df_temp['Train Loss'])
            axes[i].plot(df_temp['Model Size'], df_temp['Test Loss'])
            axes[i].set_title(f'Train Loss vs Model Size for Dataset Fraction {round(dataset_fraction,3)}')
            axes[i].set_xlabel('Model Size')
            axes[i].set_ylabel('Train Loss')
            if i == 0 and save_values is True:
                df = df_temp['Model Size'].copy()
                extract = df.to_list()
                ks.append(extract)
                df = df_temp['Train Loss'].copy()
                extract = df.to_list()
                Mlist['Train Loss'] = extract
        plt.tight_layout()
        plt.show()
    return Mlist,ks

if __name__ == "__main__":
    # Read the CSV files
    dfs1 = read_files(folder_path1,1,True)
    dfs2 = read_files(folder_path2,2,True)
    dfs3 = read_files(folder_path3,3,True)

    print("Dataframes read successfully!")
    #print(dfs1)
    #print(dfs2)
    #print(dfs3)

    # Create Mlist and ks

    Mlist,ks = create_Mlist(dfs1)
    print(Mlist,ks)
    #epochs = epochs1(dfs1, True)
    #print(epochs)

    #df = plot_test([dfs1, dfs2, dfs3],True)
    #print(df)
    
    #plot_test([dfs1], True)

    #print max and min values of each column
    #print(dfs1[0].max())
    #print(dfs1[0].min())
    #print(dfs2[0].min())
    #print(dfs2[0].max())
    #Print all unique values of Model Size

    #print(dfs1['Model Size'].unique())

    #print Train Error head()
    #print(dfs1['Train Error'].head()) 

    
"""
    #Plot model size vs the accuracy over the last 10 epochs' mean
    plt.plot(dfs1['Model Size'], dfs1['Train Accuracy'])
    plt.xlabel('Model Size')
    plt.ylabel('Accuracy')
    plt.title('Model Size vs Accuracy')
    plt.show()"""
    