import sys
import os
import pandas as pd 
import numpy as np
from gtfparse import read_gtf

import statsmodels.tsa.stattools as smt
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency

from statsmodels.stats.multitest import multipletests
from scipy.interpolate import interp1d
from scipy.integrate import odeint

import statsmodels.stats.multitest
from scipy.stats import mannwhitneyu
from statannot import add_stat_annotation

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import glob
from PIL import Image
import matplotlib.animation as animation
from multiprocessing import Pool

import random

def load_inferred_dynamics_df(data_path: str) -> pd.DataFrame:
    assert data_path.endswith(".csv"), "Data file must be a csv file"
    out_df = pd.read_csv(data_path)
    out_df.drop(columns=['Unnamed: 0', "time"], inplace=True)
    return out_df

def log2FC_transform(out_df: pd.DataFrame) -> pd.DataFrame:
    fc_df = out_df.copy()/out_df.iloc[0]
    log2FC_out_df = np.log2(fc_df)
    return log2FC_out_df

def coding_gtf_df_load(gtf_path: str) -> pd.DataFrame:
    gtf_df = read_gtf(gtf_path)
    # drop duplicates
    gtf_df = gtf_df.drop_duplicates(subset="gene_name", keep="first")
    # select columns
    gtf_df = gtf_df[['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
                     'gene_type', 'gene_name',  'transcript_name',
                     'transcript_id', 'transcript_type']]
    gtf_coding_df = gtf_df[(gtf_df["feature"] == "gene")&(gtf_df["gene_type"] == "protein_coding")]
    del gtf_df
    gtf_coding_df["start_kb"] = gtf_coding_df["start"]/1000
    gtf_coding_df["end_kb"] = gtf_coding_df["end"]/1000
    gtf_coding_df["gene_length_kb"] = gtf_coding_df["end_kb"] - gtf_coding_df["start_kb"]
    gtf_coding_df["center_kb"] = (gtf_coding_df["end_kb"] + gtf_coding_df["start_kb"])/2
    return gtf_coding_df

def gene2loci_df(gtf_coding_df: pd.DataFrame, out_df: pd.DataFrame, chromosome: str) -> pd.DataFrame:
    expressed_genes_list = out_df.columns.tolist()
    cis_gtf_coding_df = gtf_coding_df[gtf_coding_df["seqname"] == chromosome]
    cis_gtf_coding_list = cis_gtf_coding_df["gene_name"].tolist()
    expressed_cis_gtf_coding_list = list(set(cis_gtf_coding_list) & set(expressed_genes_list))
    gene2loci_dict = {}
    for gene in expressed_cis_gtf_coding_list:
        gene2loci_dict[gene] = cis_gtf_coding_df[cis_gtf_coding_df["gene_name"] == gene]["start_kb"].values[0] # gene start position
    cis_out_df = out_df[expressed_cis_gtf_coding_list]
    cis_out_df = cis_out_df.rename(columns=gene2loci_dict).sort_index(axis=1)
    return cis_out_df, gene2loci_dict

def lag_zero_plot_corr_heatmap(cis_out_df: pd.DataFrame, dynamics_name: str, chromosome: str, start_posi_kb, end_posi_kb) -> None:
    assert start_posi_kb < end_posi_kb, "start position must be less than end position"
    assert start_posi_kb >= 0, "start position must be greater than 0"
    assert dynamics_name in ["k1", "k2", "k3", "Sp", "Un"], "dynamics_name must be k1, k2, k3, Sp, Un"
    corr = cis_out_df.corr()
    plt.figure(figsize=(7,7), dpi=300)
    sns.heatmap(corr.loc[start_posi_kb:end_posi_kb, start_posi_kb:end_posi_kb], cmap="coolwarm", vmax=1, vmin=-1)
    if dynamics_name == "k1":
        plt.title(f"Lag 0: Transcription rate(α) Cross-Corr matrix on {chromosome}")
    elif dynamics_name == "k2":
        plt.title(f"Lag 0: Splicing rate(β) Cross-Corr matrix on {chromosome}")
    elif dynamics_name == "k3":
        plt.title(f"Lag 0: Degradation rate(γ) Cross-Corr matrix on {chromosome}")
    elif dynamics_name == "Sp":
        plt.title(f"Lag 0: Spliced mRNA dynamics Cross-Corr matrix on {chromosome}")
    elif dynamics_name == "Un":
        plt.title(f"Lag 0: Unspliced mRNA dynamics Cross-Corr matrix on {chromosome}")

    #plt.title(f"Lag 0: Transcription rate(α) Cross-Corr matrix on {chromosome}")
    plt.ylabel("genomic coordinates (kb)")
    plt.xlabel("genomic coordinates (kb)")
    #plt.savefig(out_path)
    #plt.close()
    
def lag_shifted_plot_corr_heatmap(cis_out_df: pd.DataFrame, chromosome: str, start_posi_kb, end_posi_kb, max_lag: int, step: int, prefix: str) -> str:
    assert start_posi_kb < end_posi_kb, "start position must be less than end position"
    assert start_posi_kb >= 0, "start position must be greater than 0 kb"
    assert max_lag > 0, "lag must be greater than 0"
    assert step > 0, "step must be greater than 0"
    assert prefix != "", "prefix must not be empty"
    # Create a directory to save the lag-shifted plots
    current_dir = os.getcwd()
    out_dir_path = os.path.join(current_dir, f"lag-shifted_plots/{prefix}")
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path, exist_ok=True)

    # Create lag-shifted plots
    for lag in range(0, max_lag+step, step):
        zeros_lag =  "{0:03d}".format(lag)
        #print(zeros_lag)
        ccc_df = pd.DataFrame()
        target_ccc_df = pd.DataFrame()
        for target_posi in cis_out_df.loc[:, start_posi_kb:end_posi_kb].columns:
            target_ccc_df = pd.DataFrame()
            for neighbor_posi in cis_out_df.loc[:, start_posi_kb:end_posi_kb].columns:
                ccc = smt.ccf(cis_out_df[neighbor_posi], cis_out_df[target_posi], unbiased=False)[lag]
                ccc = pd.Series(ccc, name=f"{neighbor_posi}")
                target_ccc_df = pd.concat([target_ccc_df, ccc], axis=1)
            target_ccc_df.index = [target_posi]
            ccc_df = pd.concat([ccc_df, target_ccc_df], axis=0)
        fig, (ax1, ax2) = plt.subplots(figsize=(6,7), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 14]}, tight_layout=True, dpi=200)
        sns.barplot(x=[lag], y=["Lag (min)"], ax=ax1, color="black", linewidth=0.5)
        ax1.set_xlim(0,300)
        sns.heatmap(ccc_df, cmap="coolwarm", vmax=1, vmin=-1, center=0.0, ax=ax2)
        # set y-axis rotation
        plt.yticks(rotation=0)

        out_file_path = os.path.join(out_dir_path, f"lag_{zeros_lag}.png")
        plt.savefig(out_file_path)
        plt.close()
    return out_dir_path

def lag_shifted_plot_corr_heatmap_select_y_axis(cis_out_df: pd.DataFrame, chromosome: str, start_posi_kb, end_posi_kb, max_lag: int, step: int, prefix: str, cis_dict:dict, target_start=None, target_end=None) -> str:
    assert start_posi_kb < end_posi_kb, "start position must be less than end position"
    assert start_posi_kb >= 0, "start position must be greater than 0 kb"
    assert max_lag > 0, "lag must be greater than 0"
    assert step > 0, "step must be greater than 0"
    assert prefix != "", "prefix must not be empty"
    assert cis_dict != {}, "cis_dict must not be empty"
    if target_start is None:
        target_start = start_posi_kb
    else:
        assert target_start >= start_posi_kb, "target_start must be greater than start_posi_kb"
    if target_end is None:
        target_end = end_posi_kb
    else:
        assert target_end <= end_posi_kb, "target_end must be less than end_posi_kb"
    # Create a directory to save the lag-shifted plots
    current_dir = os.getcwd()
    out_dir_path = os.path.join(current_dir, f"lag-shifted_plots/{prefix}")
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path, exist_ok=True)

    # Create lag-shifted plots
    for lag in range(0, max_lag+step, step):
        zeros_lag =  "{0:03d}".format(lag)
        #print(zeros_lag)
        ccc_df = pd.DataFrame()
        target_ccc_df = pd.DataFrame()
        for target_posi in cis_out_df.loc[:, start_posi_kb:end_posi_kb].columns:
            if (target_posi <= target_start) or (target_posi >= target_end):
                print(f"target_posi: {target_posi} is out of range")
                continue
            else:
                target_ccc_df = pd.DataFrame()
                for neighbor_posi in cis_out_df.loc[:, start_posi_kb:end_posi_kb].columns:
                    ccc = smt.ccf(cis_out_df[neighbor_posi], cis_out_df[target_posi], unbiased=False)[lag]
                    #ccc = pd.Series(ccc, name=f"{neighbor_posi}")
                    ccc = pd.Series(ccc, name=cis_dict[neighbor_posi])
                    target_ccc_df = pd.concat([target_ccc_df, ccc], axis=1)
                #target_ccc_df.index = [target_posi]
                target_ccc_df.index = [cis_dict[target_posi]]
                ccc_df = pd.concat([ccc_df, target_ccc_df], axis=0)
            fig, (ax1, ax2) = plt.subplots(figsize=(6,3), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 14]}, tight_layout=True, dpi=200)
            sns.barplot(x=[lag], y=["Lag (min)"], ax=ax1, color="black", linewidth=0.5)
            ax1.set_xlim(0,300)
            sns.heatmap(ccc_df, cmap="coolwarm", vmax=1, vmin=-1, center=0.0, ax=ax2)
            # set y-axis rotation
            plt.yticks(rotation=0)

            out_file_path = os.path.join(out_dir_path, f"lag_{zeros_lag}.png")
            plt.savefig(out_file_path)
            plt.close()
    return out_dir_path

def make_animation(out_dir_path: str, ms_per_frame: int, prefix: str) -> None:
    assert ms_per_frame > 0, "ms_per_frame must be greater than 0"
    assert prefix != "", "prefix must not be empty"
    # Create an animation
    files = sorted(glob.glob(out_dir_path + "/*.png"))
    image_array = []

    for my_file in files:
        
        image = Image.open(my_file)
        image_array.append(image)
    # Create the figure and axes objects
    fig, ax = plt.subplots(1,1, dpi=300)
    #plt.subplots_adjust(top=1)
    ax.axis('off')

    # Set the initial image
    im = ax.imshow(image_array[-1], animated=True)

    def update(i):
        im.set_array(image_array[i])
        return im, 
    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames = len(image_array), interval = ms_per_frame, blit= False, repeat_delay = 1000, )

    # Create a directory to save the lag-shifted plots
    current_dir = os.getcwd()
    out_dir_path = os.path.join(current_dir, "animation_gif")
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path, exist_ok=True)

    save_path = os.path.join(out_dir_path, f"{prefix}_animation.gif")
    animation_fig.save(save_path, writer="pillow")
    plt.close()

def plot_expressed_genes_enhancer_loci(gtf_df, enhancer_df, chrom, start_loci_kb:int, end_loci_kb:int, margin=None, figsize=(10, 4), cmap_list: list = None):
    chrom_gtf = gtf_df[gtf_df["seqname"] == chrom]
    chrom_enhancer = enhancer_df[enhancer_df["chr"] == chrom]

    assert end_loci_kb > start_loci_kb, "end_loci_kb must be greater than start_loci_kb"
    if margin is None:
        margin = 50
    else:
        assert margin > 0, "margin must be greater than 0"
    

    chrom_gtf = chrom_gtf[(chrom_gtf["start_kb"] >= start_loci_kb) & (chrom_gtf["end_kb"] <= end_loci_kb)]
    chrom_gtf.reset_index(drop=True, inplace=True)
    print(chrom_gtf)
    chrom_enhancer = chrom_enhancer[(chrom_enhancer["center_kb"] >= start_loci_kb + margin) & (chrom_enhancer["center_kb"] <= end_loci_kb + margin)]
    chrom_enhancer.reset_index(drop=True, inplace=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=300)

    if cmap_list is None:
        cmap_list = ["tab:purple"]*len(chrom_gtf)
    else:
        assert len(cmap_list) == len(chrom_gtf), "The length of cmap_list must be the same as the number of genes" 
    # plot genes region

    for i, row in chrom_gtf.iterrows():
        print(i)
        if i == 0:
            axes[0].add_patch(
                Rectangle(
                    (row["start_kb"], -1), (row["end_kb"] - row["start_kb"])*0.9, 0.3, 
                    color=cmap_list[i], label="Gene loci"
                )
            )
        else:
            axes[0].add_patch(
                Rectangle(
                    (row["start_kb"], -1), (row["end_kb"] - row["start_kb"])*0.9, 0.3, 
                    color=cmap_list[i]
                )
            )
        axes[0].text(
            (row["start_kb"] + row["end_kb"]) / 2, -0.5, row["gene_name"], 
            ha="center", va="bottom", fontsize=12, rotation=45
        )


    axes[0].set_xlim(chrom_gtf["start_kb"].min() - margin, chrom_gtf["end_kb"].max() + margin)
    axes[0].set_xlabel("genomic coordinates (kb)", fontsize=12)
    axes[0].set_ylim(-1, 2)
    axes[0].set_yticks([])
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["top"].set_visible(False)
    axes[0].set_title(f"Expressed Gene and eRNA loci on {chrom}", fontsize=20)
    axes[0].legend(fontsize=12, loc="upper left", frameon=False)
    
    # plot enhancer region
    #axes[1].hist(chrom_enhancer["center_kb"], color="darkred", alpha=0.5, label="Enhancer loci", bins=20)
    axes[1].bar(chrom_enhancer["center_kb"], np.log1p(chrom_enhancer["sum"]), color="darkred", alpha=0.5, width=3, label="Enhancer loci")
    axes[1].set_xlim(chrom_gtf["start_kb"].min() - margin, chrom_gtf["end_kb"].max() + margin)
    axes[1].set_xlabel("genomic coordinates (kb)", fontsize=12)
    axes[1].set_ylabel("log(total TPM + 1)", fontsize=10)
    axes[1].legend(fontsize=12, loc="upper left",frameon=False)

    #ax.annotate("~50kb", xy=(0, 1.6), xytext=(0, 1.8), arrowprops=dict(arrowstyle="-", lw=1), fontsize=8)
    #ax.annotate("", xy=(0, 1.8), xytext=(50, 1.8), arrowprops=dict(arrowstyle="<|-|>", lw=1))

    plt.tight_layout()
    plt.show()    

def load_FANTOM5_filtered_id_df(sample_name2library_id_path: str, search_sample_name: str) -> pd.DataFrame:
    sample2id_df = pd.read_csv(sample_name2library_id_path, sep="\t", header=None, names=["sample_name", "enhancer_id"])
    filtered_id_df = sample2id_df[sample2id_df["sample_name"].str.contains(search_sample_name, case=False, na=False)]
    return filtered_id_df

def load_FANTOM5_enhancer_df(FANTOM5_enhancer_path: str, bin_kb = None) -> pd.DataFrame:
    enhancer_df = pd.read_csv(FANTOM5_enhancer_path, sep="\t", index_col=0).reset_index()
    
    # Split 'index' column by ':' into 'chr' and 'start-end'
    split_chr_start_end = enhancer_df["index"].str.split(":", expand=True)

    # Split 'start-end' part by '-' into 'start' and 'end'
    split_start_end = split_chr_start_end[1].str.split("-", expand=True)

    # Add the split results as new columns to the original DataFrame
    enhancer_df["chr"] = split_chr_start_end[0]     # Add 'chr' column
    enhancer_df["start"] = split_start_end[0].astype(int)   # Add 'start' column and convert to int
    enhancer_df["end"] = split_start_end[1].astype(int)     # Add 'end' column and convert to int
    enhancer_df["start_kb"] = enhancer_df["start"]/1000
    enhancer_df["end_kb"] = enhancer_df["end"]/1000
    enhancer_df["enhancer_length_kb"] = enhancer_df["end_kb"] - enhancer_df["start_kb"]
    enhancer_df["center_kb"] = (enhancer_df["end_kb"] + enhancer_df["start_kb"])/2
    # Binning enhancer centers
    if bin_kb is None:
        bin_kb = 25
        enhancer_df["binned_center_kb"] = np.floor(enhancer_df["center_kb"] / bin_kb) * bin_kb
    elif type(bin_kb) is not int:
        raise TypeError("bin_kb must be an integer")
    elif bin_kb <= 0:
        raise ValueError("bin_kb must be greater than 0")
    else:
        enhancer_df["binned_center_kb"] = np.floor(enhancer_df["center_kb"] / bin_kb) * bin_kb

    return enhancer_df

def filter_FANTOM5_enhancer_df(filtered_id_df: pd.DataFrame, enhancer_df: pd.DataFrame) -> pd.DataFrame:
    enhancer_columns = filtered_id_df["enhancer_id"].to_list()
    enhancer_df["sum"] = enhancer_df[enhancer_columns].sum(axis=1)
    
    columns_to_keep = enhancer_columns + ['index','chr', 'start', 'end', 'start_kb', 'end_kb', 'enhancer_length_kb', 'center_kb', 'sum', 'binned_center_kb']
    filtered_enhancer_df = enhancer_df[columns_to_keep]
    return filtered_enhancer_df

def classify_symmetry(x):
    if x >= 0.5:
        return "symmetric"
    #elif x >= 0.5:
    #    return "> 0.5"
    # elif x > 0.6:
    #     return "> 0.6"
    # elif x > 0.4:
    #     return "> 0.4"
    #elif x > 0.25:
    #     return "> 0.25"
    else:
        return "asymmetric"
    
def classify_enhancer_density(x):
    if x >= 3:
        return "clustered"
    else:
        return "non-clustered"

def classify_distance(dist, unit_kb=50):
    if dist < unit_kb:
        return f"< {unit_kb}kb"
    elif dist < 2*unit_kb:
        return f"{unit_kb}-{2*unit_kb}kb"
    elif dist < 3*unit_kb:
        return f"{2*unit_kb}-{3*unit_kb}kb"
    elif dist < 4*unit_kb:
        return f"{3*unit_kb}-{4*unit_kb}kb"
    elif dist < 5*unit_kb:
        return f"{4*unit_kb}-{5*unit_kb}kb"
    elif dist < 6*unit_kb:
        return f"{5*unit_kb}-{6*unit_kb}kb"
    elif dist < 7*unit_kb:
        return f"{6*unit_kb}-{7*unit_kb}kb"
    elif dist < 8*unit_kb:
        return f"{7*unit_kb}-{8*unit_kb}kb"
    elif dist < 9*unit_kb:
        return f"{8*unit_kb}-{9*unit_kb}kb"
    elif dist < 10*unit_kb:
        return f"{9*unit_kb}-{10*unit_kb}kb"
    else:
        return f"over {10*unit_kb}kb"
    
def update_diff_dist(df):
    df['diff_dist'] = np.where(
        df['enhancer_loci'] == 'Lateral',
        abs(df['gene1_abs_dist'] - df['gene2_abs_dist']),
        abs(df['gene1_abs_dist'] + df['gene2_abs_dist'])
    )
    df = df[df['diff_dist'] > 10] # remove pairs with less than 10kb difference
    return df


def within_and_outside_correlation_df(out_df: pd.DataFrame, gtf_df: pd.DataFrame, cis_df: pd.DataFrame, distance_kb: int, enhancer_distance_kb: int, hvg_list: list) -> pd.DataFrame:

    # pair-wise correlation among neighboring genes
    all_corr_pairs_upper = pd.DataFrame()
    all_corr_pairs_upper_average_by_distance = pd.DataFrame()
    all_corr_pairs_upper_outside = pd.DataFrame()
    all_corr_pairs_upper_outside_average = pd.DataFrame()

    enhancer_df = cis_df
    gene_position_df = gtf_df
    dynamics_df = out_df
    expressed_genes_list = dynamics_df.columns.tolist()
    # Filtering by expressed genes
    gene_position_df = gene_position_df[gene_position_df["gene_name"].isin(expressed_genes_list)]

    # neighborhood distance (kb)
    assert distance_kb > 0, "distance_kb must be greater than 0"
    assert enhancer_distance_kb > 0, "enhancer_distance_kb must be greater than 0"
    distance = distance_kb
    enhancer_distance = enhancer_distance_kb

    for idx in enhancer_df.index:
        chromosome = enhancer_df.loc[idx, ]["chr"]
        enhancer_position = enhancer_df.loc[idx, ]["center_kb"]
        
        # filter genes in the neighbors of the enhancer
        filtered_df = gene_position_df[(gene_position_df["seqname"] == chromosome) & 
                                    (np.abs(gene_position_df["start_kb"] - enhancer_position) <= distance)]
        filtered_df["distance_to_enhancer"] = filtered_df["start_kb"] - enhancer_position

        filtered_outside_df = gene_position_df[(gene_position_df["seqname"] == chromosome) & 
                                    (np.abs(gene_position_df["start_kb"] - enhancer_position) > distance)] #outside of the enhancer scope
        filtered_outside_df["distance_to_enhancer"] = filtered_outside_df["start_kb"] - enhancer_position

        
        # Filter by gene name and calculate correlation matrix (rename gene names to distances).
        if filtered_df.shape[0] > 1 :  # Check that the filter result is not empty.
            gene_names = filtered_df["gene_name"].tolist()
            
            # Check if all gene names are present in dynamics_df
            gene_names = [gene for gene in gene_names if gene in expressed_genes_list]

            print(len(gene_names), "genes near an enhancer region")

            if len(gene_names) > 1:  # Check if enough genes are left to calculate correlations
                out_side_gene_list = filtered_outside_df["gene_name"].tolist()
                out_side_gene_list = list(set(out_side_gene_list) - set(gene_names))
                print(f"Number of neighbor genes: {len(gene_names)}, Number of outside genes: {len(out_side_gene_list)}")

                if len(out_side_gene_list) < len(gene_names):
                    print("Not enough genes in the outside region to match the number of neighbor genes. Skipping this enhancer.")
                    continue

                try:
                    random_sampled_outside_genes_list = random.sample(out_side_gene_list, len(gene_names))
                except ValueError as e:
                    print(f"Error in sampling: {e}. Skipping this enhancer.")
                    continue

                filtered_df = filtered_df[filtered_df["gene_name"].isin(gene_names)]
                filtered_outside_df = filtered_outside_df[filtered_outside_df["gene_name"].isin(random_sampled_outside_genes_list)]
                assert filtered_df.shape[0] == filtered_outside_df.shape[0], "The number of genes in the neighbors and outside of the enhancer must be the same."
                # Create a dictionary to convert 'gene_name' to 'distance_to_enhancer'.
                rename_dict = filtered_df.set_index('gene_name')['distance_to_enhancer'].to_dict()
                rename_dict_outside = filtered_outside_df.set_index('gene_name')['distance_to_enhancer'].to_dict()
           
                # Create correlation matrix and rename gene names to distances
                corr_matrix = dynamics_df[gene_names].rename(columns=rename_dict).corr()
                corr_matrix_outside = dynamics_df[random_sampled_outside_genes_list].rename(columns=rename_dict_outside).corr()

                # Create a mask to extract the upper triangle.
                mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                mask_outside = np.triu(np.ones(corr_matrix_outside.shape), k=1).astype(bool)
                upper_triangle = corr_matrix.where(mask)
                upper_triangle_outside = corr_matrix_outside.where(mask_outside)

                # Extract pairs of row names, column names and correlation values
                corr_pairs_upper = upper_triangle.stack().reset_index()
                corr_pairs_upper.columns = ['gene1_posi', 'gene2_posi', 'correlation']
                corr_pairs_upper["Enhancer loci"] = corr_pairs_upper.apply(lambda row: "Lateral" if row["gene1_posi"] * row["gene2_posi"] >= 0 else "Intergenic", axis=1)
                corr_pairs_upper["gene1_abs_dist"] = np.abs(corr_pairs_upper["gene1_posi"])
                corr_pairs_upper["gene2_abs_dist"] = np.abs(corr_pairs_upper["gene2_posi"])
                corr_pairs_upper["average_dist"] = (corr_pairs_upper["gene1_abs_dist"] + corr_pairs_upper["gene2_abs_dist"])/2
                corr_pairs_upper["max_dist"] = np.maximum(corr_pairs_upper["gene1_abs_dist"], corr_pairs_upper["gene2_abs_dist"])
                corr_pairs_upper["chromosome"] = chromosome
                corr_pairs_upper["enhancer_position"] = enhancer_position
                corr_pairs_upper["including_highly_variable_genes"] = len(set(gene_names) & set(hvg_list))
                cis_neighbor_enhancer_df = enhancer_df[(enhancer_df["chr"] == chromosome) & 
                                    (np.abs(enhancer_df["center_kb"] - enhancer_position) <= enhancer_distance)]
                enhancer_num = cis_neighbor_enhancer_df.shape[0]
                corr_pairs_upper["number_of_enhancers"] = enhancer_num
                corr_pairs_upper["enhancer_density"] = corr_pairs_upper["number_of_enhancers"].apply(classify_enhancer_density)

                #corr_pairs_upper["genomic_distance"] = corr_pairs_upper["average_dist"].apply(classify_distance)
                corr_pairs_upper["genomic_distance"] = corr_pairs_upper["max_dist"].apply(classify_distance)
                corr_pairs_upper_average_by_distance = pd.DataFrame(corr_pairs_upper.groupby("genomic_distance")["correlation"].mean().reset_index())
                corr_pairs_upper_average_by_distance.rename(columns={"correlation":"average_correlation"}, inplace=True)
                corr_pairs_upper_average_by_distance["chromosome"] = chromosome
                corr_pairs_upper_average_by_distance["enhancer_position"] = enhancer_position
                corr_pairs_upper_average_by_distance["including_highly_variable_genes"] = len(set(gene_names) & set(hvg_list))
                corr_pairs_upper_average_by_distance["number_of_enhancers"] = enhancer_num

                corr_pairs_upper_outside = upper_triangle_outside.stack().reset_index()
                corr_pairs_upper_outside.columns = ['gene1_posi', 'gene2_posi', 'correlation']
                corr_pairs_upper_outside["chromosome"] = chromosome
                corr_pairs_upper_outside["enhancer_position"] = enhancer_position
                corr_pairs_upper_outside["including_highly_variable_genes"] = len(set(gene_names) & set(hvg_list))
                corr_pairs_upper_outside["enhancer_density"] = "random"

                corr_pairs_upper_outside_average = pd.DataFrame([["random", corr_pairs_upper_outside["correlation"].mean()]], columns=["genomic_distance","average_correlation"])
                corr_pairs_upper_outside_average["chromosome"] = chromosome
                corr_pairs_upper_outside_average["enhancer_position"] = enhancer_position


                # concat results
                all_corr_pairs_upper = pd.concat([all_corr_pairs_upper, corr_pairs_upper], ignore_index=True)
                all_corr_pairs_upper_average_by_distance = pd.concat([all_corr_pairs_upper_average_by_distance, corr_pairs_upper_average_by_distance], ignore_index=True)

                all_corr_pairs_upper_outside = pd.concat([all_corr_pairs_upper_outside, corr_pairs_upper_outside], ignore_index=True)
                all_corr_pairs_upper_outside_average = pd.concat([all_corr_pairs_upper_outside_average, corr_pairs_upper_outside_average], ignore_index=True)


    return all_corr_pairs_upper, all_corr_pairs_upper_outside, all_corr_pairs_upper_average_by_distance, all_corr_pairs_upper_outside_average

def within_and_outside_average_correlation_df(out_df: pd.DataFrame, gtf_df: pd.DataFrame, cis_df: pd.DataFrame, distance_kb: int, enhancer_distance_kb: int, hvg_list: list) -> pd.DataFrame:

    # pair-wise correlation among neighboring genes
    all_corr_pairs_upper_average_by_distance = pd.DataFrame()
    all_corr_pairs_upper_outside_average = pd.DataFrame()

    enhancer_df = cis_df
    gene_position_df = gtf_df
    dynamics_df = out_df
    expressed_genes_list = dynamics_df.columns.tolist()
    # Filtering by expressed genes
    gene_position_df = gene_position_df[gene_position_df["gene_name"].isin(expressed_genes_list)]

    # neighborhood distance (kb)
    assert distance_kb > 0, "distance_kb must be greater than 0"
    assert enhancer_distance_kb > 0, "enhancer_distance_kb must be greater than 0"
    distance = distance_kb
    enhancer_distance = enhancer_distance_kb

    for idx in enhancer_df.index:
        chromosome = enhancer_df.loc[idx, ]["chr"]
        enhancer_position = enhancer_df.loc[idx, ]["center_kb"]
        
        # filter genes in the neighbors of the enhancer
        filtered_df = gene_position_df[(gene_position_df["seqname"] == chromosome) & 
                                    (np.abs(gene_position_df["start_kb"] - enhancer_position) <= distance)]
        filtered_df["distance_to_enhancer"] = filtered_df["start_kb"] - enhancer_position

        filtered_outside_df = gene_position_df[(gene_position_df["seqname"] == chromosome) & 
                                    (np.abs(gene_position_df["start_kb"] - enhancer_position) > distance)] #outside of the enhancer scope
        filtered_outside_df["distance_to_enhancer"] = filtered_outside_df["start_kb"] - enhancer_position

        
        # Filter by gene name and calculate correlation matrix (rename gene names to distances).
        if filtered_df.shape[0] > 1 :  # Check that the filter result is not empty.
            gene_names = filtered_df["gene_name"].tolist()
            
            # Check if all gene names are present in dynamics_df
            gene_names = [gene for gene in gene_names if gene in expressed_genes_list]

            #print(len(gene_names), "genes near an enhancer region")

            if len(gene_names) > 1:  # Check if enough genes are left to calculate correlations
                out_side_gene_list = filtered_outside_df["gene_name"].tolist()
                out_side_gene_list = list(set(out_side_gene_list) - set(gene_names))
                print("chromosome:", chromosome, " ", "enhancer_position:", enhancer_position)
                #print(f"Number of neighbor genes: {len(gene_names)}, Number of outside genes: {len(out_side_gene_list)}")

                if len(out_side_gene_list) < len(gene_names):
                    #print("Not enough genes in the outside region to match the number of neighbor genes. Skipping this enhancer.")
                    continue

                try:
                    random_sampled_outside_genes_list = random.sample(out_side_gene_list, len(gene_names))
                except ValueError as e:
                    #print(f"Error in sampling: {e}. Skipping this enhancer.")
                    continue

                filtered_df = filtered_df[filtered_df["gene_name"].isin(gene_names)]
                filtered_outside_df = filtered_outside_df[filtered_outside_df["gene_name"].isin(random_sampled_outside_genes_list)]
                assert filtered_df.shape[0] == filtered_outside_df.shape[0], "The number of genes in the neighbors and outside of the enhancer must be the same."
                # Create a dictionary to convert 'gene_name' to 'distance_to_enhancer'.
                rename_dict = filtered_df.set_index('gene_name')['distance_to_enhancer'].to_dict()
                rename_dict_outside = filtered_outside_df.set_index('gene_name')['distance_to_enhancer'].to_dict()
           
                # Create correlation matrix and rename gene names to distances
                corr_matrix = dynamics_df[gene_names].rename(columns=rename_dict).corr()
                corr_matrix_outside = dynamics_df[random_sampled_outside_genes_list].rename(columns=rename_dict_outside).corr()

                # Create a mask to extract the upper triangle.
                mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                mask_outside = np.triu(np.ones(corr_matrix_outside.shape), k=1).astype(bool)
                upper_triangle = corr_matrix.where(mask)
                upper_triangle_outside = corr_matrix_outside.where(mask_outside)

                # Extract pairs of row names, column names and correlation values
                corr_pairs_upper = upper_triangle.stack().reset_index()
                corr_pairs_upper.columns = ['gene1_posi', 'gene2_posi', 'correlation']
                corr_pairs_upper["enhancer_loci"] = corr_pairs_upper.apply(lambda row: "Lateral" if row["gene1_posi"] * row["gene2_posi"] >= 0 else "Intergenic", axis=1)
                corr_pairs_upper["gene1_abs_dist"] = np.abs(corr_pairs_upper["gene1_posi"])
                corr_pairs_upper["gene2_abs_dist"] = np.abs(corr_pairs_upper["gene2_posi"])
                corr_pairs_upper["average_dist"] = (corr_pairs_upper["gene1_abs_dist"] + corr_pairs_upper["gene2_abs_dist"])/2
                corr_pairs_upper["max_dist"] = np.maximum(corr_pairs_upper["gene1_abs_dist"], corr_pairs_upper["gene2_abs_dist"])
                corr_pairs_upper["diff_dist"] = np.abs(corr_pairs_upper["gene1_abs_dist"] - corr_pairs_upper["gene2_abs_dist"])
                corr_pairs_upper = update_diff_dist(corr_pairs_upper)

                cis_neighbor_enhancer_df = enhancer_df[(enhancer_df["chr"] == chromosome) & 
                                    (np.abs(enhancer_df["center_kb"] - enhancer_position) <= enhancer_distance)]
                enhancer_num = cis_neighbor_enhancer_df.shape[0]

                #corr_pairs_upper["genomic_distance"] = corr_pairs_upper["average_dist"].apply(classify_distance)
                corr_pairs_upper["genomic_distance"] = corr_pairs_upper["max_dist"].apply(classify_distance)

                corr_pairs_upper_average_by_distance = pd.DataFrame(corr_pairs_upper.groupby("genomic_distance")["correlation"].mean().reset_index())
                corr_pairs_upper_average_by_distance.rename(columns={"correlation":"average_correlation"}, inplace=True)
                corr_pairs_upper_average_by_distance["chromosome"] = chromosome
                corr_pairs_upper_average_by_distance["enhancer_position"] = enhancer_position
                corr_pairs_upper_average_by_distance["including_highly_variable_genes"] = len(set(gene_names) & set(hvg_list))
                corr_pairs_upper_average_by_distance["number_of_enhancers"] = enhancer_num

                corr_pairs_upper_outside = upper_triangle_outside.stack().reset_index()
                corr_pairs_upper_outside.columns = ['gene1_posi', 'gene2_posi', 'correlation']

                corr_pairs_upper_outside_average = pd.DataFrame([["random", corr_pairs_upper_outside["correlation"].mean()]], columns=["genomic_distance","average_correlation"])
                corr_pairs_upper_outside_average["chromosome"] = chromosome
                corr_pairs_upper_outside_average["enhancer_position"] = enhancer_position


                del corr_pairs_upper, corr_pairs_upper_outside, cis_neighbor_enhancer_df
                # concat results
                all_corr_pairs_upper_average_by_distance = pd.concat([all_corr_pairs_upper_average_by_distance, corr_pairs_upper_average_by_distance], ignore_index=True)

                all_corr_pairs_upper_outside_average = pd.concat([all_corr_pairs_upper_outside_average, corr_pairs_upper_outside_average], ignore_index=True)

                del corr_pairs_upper_average_by_distance, corr_pairs_upper_outside_average

    return all_corr_pairs_upper_average_by_distance, all_corr_pairs_upper_outside_average

def load_AURA_data(cis_file_path, trans_file_path, species="human"):
    # Load AURA data
    AURA_cis_df = pd.read_csv(cis_file_path, sep="\t")
    AURA_trans_df = pd.read_csv(trans_file_path, sep="\t")

    # Filtering by species
    if species == "human":
        AURA_cis_df = AURA_cis_df[AURA_cis_df["Target UTR"].str.contains("hg19")]
        AURA_trans_df = AURA_trans_df[AURA_trans_df["Target UTR"].str.contains("hg19")]
    elif species == "mouse":
        AURA_cis_df = AURA_cis_df[AURA_cis_df["Target UTR"].str.contains("mm10")]
        AURA_trans_df = AURA_trans_df[AURA_trans_df["Target UTR"].str.contains("mm10")]
    else:
        raise ValueError("species must be either 'human' or 'mouse'")
    
    # 5'utr, 3'utr separated dataframes
    AURA_cis_5UTR_df = AURA_cis_df[AURA_cis_df["Target UTR"].str.contains("5UTR")]
    AURA_cis_3UTR_df = AURA_cis_df[AURA_cis_df["Target UTR"].str.contains("3UTR")]
    AURA_trans_5UTR_df = AURA_trans_df[AURA_trans_df["Target UTR"].str.contains("5UTR")]
    AURA_trans_3UTR_df = AURA_trans_df[AURA_trans_df["Target UTR"].str.contains("3UTR")]

    return AURA_cis_3UTR_df, AURA_trans_3UTR_df, AURA_cis_5UTR_df, AURA_trans_5UTR_df

def extract_cis_trans_list(AURA_cis_3UTR_df, AURA_trans_3UTR_df, AURA_cis_5UTR_df, AURA_trans_5UTR_df):
    # Extract features  cis elements and trans factors
    cis_3UTR_list = AURA_cis_3UTR_df["Cis-element"].unique().tolist()
    trans_3UTR_list = AURA_trans_3UTR_df["Regulatory factor"].unique().tolist()
    cis_5UTR_list = AURA_cis_5UTR_df["Cis-element"].unique().tolist()
    trans_5UTR_list = AURA_trans_5UTR_df["Regulatory factor"].unique().tolist()
    
    return cis_3UTR_list, trans_3UTR_list, cis_5UTR_list, trans_5UTR_list

def trans_factor_enrichment(trans_factor, AURA_trans_df, target_genes, background_genes):
    """
    Calculates the contingency table for trans factor enrichment analysis.
    """
    known_trans_target_set = set(AURA_trans_df[AURA_trans_df["Regulatory factor"] == trans_factor]["Target Gene"].unique().tolist())
    
    trans_target_num = len(known_trans_target_set & set(target_genes))
    trans_background_num = len(known_trans_target_set & set(background_genes))
    target_non_trans_target_num = len(set(target_genes)) - trans_target_num
    background_non_trans_target_num = len(set(background_genes)) - trans_background_num
    
    data_mat = [[trans_target_num, trans_background_num], 
                [target_non_trans_target_num, background_non_trans_target_num]] # contingency table 
    return data_mat

def cis_element_enrichment(cis_element, AURA_cis_df, target_genes, background_genes):
    """
    Calculates the contingency table for cis element enrichment analysis.
    """
    known_cis_target_set = set(AURA_cis_df[AURA_cis_df["Cis-element"] == cis_element]["Target Gene"].unique().tolist())
    
    cis_target_num = len(known_cis_target_set & set(target_genes))
    cis_background_num = len(known_cis_target_set & set(background_genes))
    target_non_cis_target_num = len(set(target_genes)) - cis_target_num
    background_non_cis_target_num = len(set(background_genes)) - cis_background_num
    
    data_mat = [[cis_target_num, cis_background_num], 
                [target_non_cis_target_num, background_non_cis_target_num]] # contingency table 
    return data_mat


def perform_cross_table_test(data_mat, test_type='fisher', alternative=None):
    """
    Performs a statistical test (Fisher's exact test or chi-square test) on the given data matrix and returns the p-value.
    """
    if alternative is None:
        alternative = 'greater'
    elif alternative not in ['greater', 'less', 'two-sided']:
        raise ValueError("Invalid alternative. Choose 'greater', 'less', or 'two-sided'.")
    if test_type == 'fisher':
        odds_r, p_value = fisher_exact(data_mat, alternative=alternative)
        return p_value, odds_r
    elif test_type == 'chi2':
        _, p_value, _, expected_freq = chi2_contingency(data_mat, correction=False)
        return p_value, expected_freq
    else:
        raise ValueError("Invalid test_type. Choose 'fisher' or 'chi2'.")
    


def adjustesd_statistical_test_for_trans_factors(expressed_trans_factors_list, AURA_trans_df, target_genes, background_genes, test_type='fisher', alternative=None):

    results = []

    for trans_factor in expressed_trans_factors_list:
        print(trans_factor)
        data_mat = trans_factor_enrichment(trans_factor, AURA_trans_df, target_genes, background_genes)
        if test_type == 'fisher':
            p_value, Odds_ratio = perform_cross_table_test(data_mat, test_type='fisher', alternative=alternative)
            #print("Scipy_OR: ", Odds_ratio)
        elif test_type == 'chi2':
            p_value, expected_freq = perform_cross_table_test(data_mat, test_type='chi2')
            #print(expected_freq)
        results.append({"trans-factor": trans_factor, "p-value": p_value, "odds_ratio": Odds_ratio, "target_num": data_mat[0][0]})

    enrichment_df = pd.DataFrame(results)

    # Correct p-values for multiple testing using the Benjamini-Hochberg method
    _, corrected_pvalues, _, _ = multipletests(enrichment_df["p-value"], method='fdr_bh')
    enrichment_df["adjusted p-value"] = corrected_pvalues
    enrichment_df["-log10(p-value)"] = -np.log10(enrichment_df["p-value"])
    enrichment_df["-log10(adjusted p-value)"] = -np.log10(enrichment_df["adjusted p-value"])
    
    return enrichment_df

def adjustesd_statistical_test_for_cis_elements(expressed_cis_elements_list, AURA_cis_df, target_genes, background_genes, test_type='fisher', alternative=None):

    results = []

    for cis_element in expressed_cis_elements_list:
        print(cis_element)
        data_mat = cis_element_enrichment(cis_element, AURA_cis_df, target_genes, background_genes)

        if test_type == 'fisher':
            p_value = perform_cross_table_test(data_mat, test_type='fisher', alternative=alternative)
        elif test_type == 'chi2':
            p_value, expected_freq = perform_cross_table_test(data_mat, test_type='chi2')
        results.append({"cis-element": cis_element, "p-value": p_value, "odds_ratio": Odds_ratio, "target_num": data_mat[0][0]})

    enrichment_df = pd.DataFrame(results)

    # Correct p-values for multiple testing using the Benjamini-Hochberg method
    _, corrected_pvalues, _, _ = multipletests(enrichment_df["p-value"], method='fdr_bh')
    enrichment_df["adjusted p-value"] = corrected_pvalues
    enrichment_df["-log10(p-value)"] = -np.log10(enrichment_df["p-value"])
    enrichment_df["-log10(adjusted p-value)"] = -np.log10(enrichment_df["adjusted p-value"])
    
    return enrichment_df

def adjusted_Mann_Whitney(data, box_pairs, x, y, hue, alpha=0.05, correction_method=None, alternative=None):
    """
    Perform Mann-Whitney U tests for multiple pairs and apply multiple testing correction.   
    Returns:
        dict: A dictionary containing:
            - pairs: List of all tested pairs.
            - pvalues: List of raw p-values.
            - adjusted_pvalues: List of corrected p-values.
            - significant_pairs: List of significant pairs after correction.
    """
    results = {"pairs": [], "pvalues": [], "adjusted_pvalues": [], "significant_pairs": []}

    # Set default correction method to 'bonferroni' if not provided
    if correction_method is None:
        correction_method = "bonferroni"
    else:
        assert correction_method in ["bonferroni", "fdr_bh"], \
            "correction_method must be 'bonferroni' or 'fdr_bh'."

    # Set default alternative hypothesis to 'two-sided' if not provided
    if alternative is None:
        alternative = "two-sided"
        print(f"null hypothesis: Central Tendency of Group1 is equal to Group2")
    else:
        assert alternative in ["two-sided", "less", "greater"], \
            "alternative must be 'two-sided', 'less', or 'greater'."
        print(f"null hypothesis: Central Tendency of Group1 is {alternative} than Group2")
    
    # Perform Mann-Whitney U tests for each pair
    for pair in box_pairs:
        # Extract data for each group in the pair
        group1 = data[(data[x] == pair[0][0]) & (data[hue] == pair[0][1])][y]
        group2 = data[(data[x] == pair[1][0]) & (data[hue] == pair[1][1])][y]

        print(f"Group1:{pair[0][0]}_&_{pair[0][1]}, Group2:{pair[1][0]}_&_{pair[1][1]}")

        # Perform Mann-Whitney U test
        _, p_value = mannwhitneyu(group1, group2, alternative=alternative)

        # Store results
        results["pairs"].append(pair)
        results["pvalues"].append(p_value)
    
    # Apply multiple testing correction
    _, corrected_pvalues, _, _ = statsmodels.stats.multitest.multipletests(
        results["pvalues"], alpha=alpha, method=correction_method
    )
    results["adjusted_pvalues"] = corrected_pvalues

    # Filter significant pairs
    for pair, adjusted_p in zip(results["pairs"], results["adjusted_pvalues"]):
        if adjusted_p < alpha:
            results["significant_pairs"].append(pair)

    return results


def compute_ccf(gene, x_df, y_df):
    """Compute cross-correlation for a single gene. [x_k:x_n[] vs [y_0:y_n-k]"""
    ccf = smt.ccf(x_df[gene], y_df[gene], unbiased=False)
    return pd.Series(ccf, name=f"{gene}")

def compute_ccf_pair_gene(x_df, y_df, x_gene, y_gene):
    """Compute cross-correlation for a pair gene. [x_k:x_n[] vs [y_0:y_n-k]"""
    ccf = smt.ccf(x_df[x_gene], y_df[y_gene], unbiased=False)
    return pd.Series(ccf, name=f"Cross-Corr:{y_gene}_vs_{x_gene}")

def lag_shifted_cross_corr_multiprocessing(x_df: pd.DataFrame, y_df: pd.DataFrame, num_cores: int = None) -> pd.DataFrame:
    assert x_df.shape == y_df.shape, "The input and output dataframes should have the same shape"

    # Use a conservative number of cores: maximum available minus 1
    if num_cores is None:
        num_cores = max(1, os.cpu_count() - 1)  # Ensure at least 1 core is used

    gene_list = x_df.columns.tolist()
    
    # Use multiprocessing Pool to parallelize the computation
    with Pool(processes=num_cores) as pool:
        # Pass x_df and y_df as additional arguments to compute_ccf
        ccc_series_list = pool.starmap(compute_ccf, [(gene, x_df, y_df) for gene in gene_list])
    
    # Concatenate all the Series objects into a single DataFrame
    ccc_df = pd.concat(ccc_series_list, axis=1)
    return ccc_df

def plot_kinetic_dynamics(
    gene,                     # Gene name to be plotted
    time_data,                # Time data (x-axis)
    mean_data,                # DataFrame with mean values
    std_data,                 # DataFrame with standard deviation values
    y_label = "kinetic", # Label for the y-axis (default value)
    line_color='k',           # Line color
    fill_color=(0.1, 0.2, 0.5, 0.3), # Fill color for uncertainty region (RGBA)
    line_width=5,             # Line width
    x_tick_size=40,           # Font size for x-axis ticks
    y_tick_size=50,           # Font size for y-axis ticks
    font_size=30,             # General font size for the plot
    ylabel_size=80,           # Font size for the y-axis label
    fig_size=(25, 12.5)       # Figure size (width, height)
):
    # Initialize the figure and axes
    fig, ax = plt.subplots()
    
    # Plot the mean transcription rate
    ax.plot(time_data, mean_data[gene], color=line_color, lw=line_width)
    
    # Fill the uncertainty region (mean ± standard deviation)
    ax.fill_between(
        time_data,
        mean_data[gene] + std_data[gene],
        mean_data[gene] - std_data[gene],
        facecolor=fill_color,
        interpolate=True
    )
    
    # Customize tick labels for x and y axes
    ax.tick_params(axis='x', labelsize=x_tick_size)
    ax.tick_params(axis='y', labelsize=y_tick_size)
    plt.rc('font', size=font_size)  # Set the general font size
    
    # Remove the right and top spines for a cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set the y-axis label
    ax.set_ylabel(y_label, fontsize=ylabel_size)
    ax.set_ylim(0, )  # Ensure the y-axis starts
    
    # Set the figure size
    fig.set_size_inches(*fig_size)
    
    return fig, ax 

def load_inspect_csv(path, file_name):
    replace_map = {
    "inspect_k1.csv": "synthesis_",
    "inspect_k2.csv": "processing_",
    "inspect_k3.csv": "degradation_",
    "total.csv": "total_",
    "premrna.csv": "preMRNA_"}
    df = pd.read_csv(f"{path}/{file_name}", index_col=0)
    replace_str = replace_map[file_name]
    df.columns = df.columns.str.replace(replace_str, '', regex=False).astype(float)
    df = df.T
    df.dropna(axis=1, inplace=True) # Drop columns with NaN values
    return df

def interpolated_df(df, time_range_minutes):
    interpolated_df = pd.DataFrame()
    time_hours = df.index.to_numpy()  # Index is in hours

    for col in df.columns:
        values = df[col].to_numpy()
        # Create a linear interpolation function
        linear_interp = interp1d(time_hours, values, kind='linear')
        # Compute interpolated values on a per-minute basis
        interpolated_values = linear_interp(time_range_minutes)
        interpolated_df[col] = interpolated_values

    # Set the per-minute index
    interpolated_df.index = (time_range_minutes*60).astype(int) # minutes 
    return interpolated_df

def reshape_df(true, pred):
    assert true.shape == pred.shape, "The shapes of the two arrays must be the same."
    return true.reshape(-1,), pred.reshape(-1,)

def lagged_correlation(true_values, predicted_values, lag: int):
    assert len(true_values) == len(predicted_values), "The lengths of the two time series must be the same."
    assert lag >= 0, "The lag must be a non-negative integer."
    return smt.ccf(true_values, predicted_values, adjusted=False)[lag]

