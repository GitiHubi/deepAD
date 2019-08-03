# import class libraries
import os as os
import json as js
import numpy as np
import datetime as dt
import glob
import argparse
import torch
import pandas as pd

# class Utilities
class UtilsHandler(object):

    def __init__(self):
        pass

    def create_new_sub_directory(self, timestamp, parent_dir, indicator):

        # create sub directory name
        sub_dir = os.path.join(parent_dir, str(timestamp) + '_' + str(indicator) + '_experiment')

        # case experiment sub directory does not exist
        if not os.path.exists(sub_dir):
            # create new sub directory
            os.makedirs(sub_dir)

        # return new sub directory
        return sub_dir

    def create_experiment_sub_directory(self, parent_dir, folder_name):

        # create sub directory name
        sub_dir = os.path.join(parent_dir, folder_name)

        # case experiment sub directory does not exist
        if (not os.path.exists(sub_dir)):
            # create new sub directory
            os.makedirs(sub_dir)

        # return new sub directory
        return sub_dir

    def create_experiment_directory(self, param, parent_dir, architecture='attention'):

        # case: base architecture
        if architecture == 'base':

            # create experiment directory name
            experiment_directory_name = '{}_base_exp_{}_{}_isd_{}_lt_{}_dt_{}_eh_{}_dh_{}_la_{}_ts_{}_lr_{}_op_{}_dc_{}_osd_{}_ep_{}-{}'.format(
                str(param['exp_timestamp']), str(param['market']), str(param['indicator']), str(param['seed']),
                str(param['loss_type']), str(param['is_start_date']), str(param['encoder_hidden_size']),
                str(param['decoder_hidden_size']), str(param['layer']), str(param['time_steps']),
                str(param['learning_rate']), str(param['optimizer']), str(param['learning_rate_decay']),
                str(param['is_end_date']), str(param['no_epochs']), str(param['resolution']))

        # case: plain vanilla gan architecture
        elif architecture == 'gan':

            # create experiment directory name
            experiment_directory_name = '{}_deepGAN_exp_sd_{}_ep_{}_gs_{}_rd_{}_sd_{}_mb_{}_eco_{}_gpu_{}'.format(
                str(param['exp_timestamp']), str(param['seed']), str(param['no_epochs']), str(param['no_gauss']),
                str(param['radi_gauss']), str(param['stdv_gauss']), str(param['mini_batch_size']),
                str(param['enc_output']), str(param['use_cuda']))

        # create experiment directory name
        exp_dir = os.path.join(parent_dir, experiment_directory_name)

        # case experiment directory does not exist
        if (not os.path.exists(exp_dir)):
            # create new experiment directory
            os.makedirs(exp_dir)

        # create meta data, signal data, and backtest data sub directories
        par_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='00_param')
        sta_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='01_statistics')
        enc_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='02_encodings')
        vis_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='03_visuals')
        mod_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='04_models')
        prd_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='05_predictions')

        _ = self.create_experiment_sub_directory(parent_dir=os.path.join(exp_dir, '01_statistics'),
                                                 folder_name='01_tensorboard')

        # return new experiment directory and sub directories
        return exp_dir, par_sub_dir, sta_sub_dir, enc_sub_dir, vis_sub_dir, mod_sub_dir, prd_sub_dir

    def save_experiment_parameter(self, param, parameter_dir):

        # create filename
        filename = str(
            '{}_deepGAN_exp_sd_{}_ep_{}_gs_{}_rd_{}_sd_{}_mb_{}_eco_{}_gpu_{}.txt'.format(str(param['exp_timestamp']),
                                                                                          str(param['seed']),
                                                                                          str(param['no_epochs']),
                                                                                          str(param['no_gauss']),
                                                                                          str(param['radi_gauss']),
                                                                                          str(param['stdv_gauss']),
                                                                                          str(param['mini_batch_size']),
                                                                                          str(param['enc_output']),
                                                                                          str(param['use_cuda'])))

        # write experimental config to file
        with open(os.path.join(parameter_dir, filename), 'w') as outfile:
            # dump experiment parameters
            js.dump(param, outfile)

    def read_experiment_parameter(self, experiment_dir):

        # determine experiment parameter file
        param_file_path = sorted(glob.glob(os.path.join(experiment_dir, '00_param', '*_parameter.txt')))[0]

        # init experiment parameter
        experiment_parameter = []

        # read json parameter file
        with open(param_file_path) as parameter_file:
            # iterate over json file lines
            for parameter_line in parameter_file:
                # read json parameter file line
                experiment_parameter = js.loads(parameter_line)

        # return experiment parameter
        return experiment_parameter

    def get_network_parameter(self, net):

        # init number of parameters
        num_params = 0

        # iterate over net parameters
        for param in net.parameters():
            # collect number of parameters
            num_params += param.numel()

        # return number of network parameters
        return num_params

    def str2bool(self, value):

        # case: already boolean
        if type(value) == bool:

            # return actual boolean
            return value

        # convert value to lower cased string
        # case: true acronyms
        elif value.lower() in ('yes', 'true', 't', 'y', '1'):

            # return true boolean
            return True

        # case: false acronyms
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):

            # return false boolean
            return False

        # case: no valid acronym detected
        else:

            # rais error
            raise argparse.ArgumentTypeError('[ERROR] Boolean value expected.')

    # sample Guassian distribution(s) of a particular shape
    def get_target_distribution(self, mu, sigma, n, dim=2):

        # determine samples per gaussian
        samples_per_gaussian = int(n / len(mu))

        # determine sample delta
        samples_delta = n - (len(mu) * samples_per_gaussian)

        # init samples array
        z_samples_all = np.array([])

        # iterate over all gaussians
        for i, mean in enumerate(mu):

            # case: first gaussian
            if i == 0:

                # randomly sample from gaussion distribution + samples delta
                z_samples_all = np.random.normal(mean, sigma, size=(samples_per_gaussian + samples_delta, dim))

            # case: non-first gaussian
            else:

                # randomly sample from gaussion distribution + samples delta
                z_samples = np.random.normal(mean, sigma, size=(samples_per_gaussian, dim))

                # stack new samples
                z_samples_all = np.vstack([z_samples_all, z_samples])

        # return sampled data
        return z_samples_all

    # sample Guassian distribution(s) of a particular shape
    def get_info_target_distribution(self, mu, sigma, n, dim=2):

        # determine samples per gaussian
        samples_per_gaussian = int(n / len(mu))

        # determine sample delta
        samples_delta = n - (len(mu) * samples_per_gaussian)

        # iterate over all gaussians
        for i, mean in enumerate(mu):

            # case: first gaussian
            if i == 0:

                # randomly sample from gaussion distribution + samples delta
                z_discrete_samples_all = np.zeros((samples_per_gaussian + samples_delta, len(mu)))
                z_discrete_samples_all[:, i] = 1

                # randomly sample from gaussion distribution + samples delta
                z_continous_samples_all = np.random.normal(mean, sigma, size=(samples_per_gaussian + samples_delta, dim))

            # case: non-first gaussian
            else:

                # randomly sample from gaussion distribution + samples delta
                z_discrete_samples = np.zeros((samples_per_gaussian, len(mu)))
                z_discrete_samples[:, i] = 1

                # stack new samples
                z_discrete_samples_all = np.vstack([z_discrete_samples_all, z_discrete_samples])

                # randomly sample from gaussion distribution + samples delta
                z_continous_samples = np.random.normal(mean, sigma, size=(samples_per_gaussian, dim))

                # stack new samples
                z_continous_samples_all = np.vstack([z_continous_samples_all, z_continous_samples])

        # return sampled data
        return z_discrete_samples_all, z_continous_samples_all

    # split matrix: one-hot and numerical parts
    def split_2_dummy_numeric(self, m, slice_index):

        dummy_part = m[:, :slice_index]
        numeric_part = m[:, slice_index:]

        return dummy_part, numeric_part

    def sample_continuous(self, mean, sigma, size):
        return np.random.normal(mean, sigma, size=size)

    def sample_discrete(self, size):
        return np.random.multinomial(1, [1.0/size[1]]*size[1], size=size[0])

    def compute_accuracy(self, y_pred, y_true):
        return (y_pred == y_true.long()).sum().double() / y_true.shape[0]

    def kl_div_cost(self, z):

        z_mu = torch.mean(z, dim=0)
        z_sigma = torch.std(z, dim=0)

        kl_loss = 0.5*torch.sum(torch.exp(z_sigma) + z_mu**2 - 1.0 - z_sigma)
        return kl_loss

    def augument_transactions(self, discrete_dataset, continuous_set, discrete_attribute_names):

        augumented_dataset = pd.DataFrame()

        # combine discrete and continuous datasets
        combined_dataset = pd.concat([discrete_dataset, continuous_set], axis=1)

        # get encoded attr names
        # encoded_attr_names = discrete_dataset.columns

        # augment discrete encodings
        for attribute_name in discrete_attribute_names:

            # subset of encoded dataset assigned to particular feature value
            encoded_subset_attr_names = discrete_dataset.columns[discrete_dataset.columns.str.contains(attribute_name)]

            # select the most frequent feature value in the subset
            selected_feature = combined_dataset[encoded_subset_attr_names].sum().idxmax()

            # make a copy of discrete dataset
            combined_dataset_to_update = combined_dataset.copy()
            # set subset of attributes to zero
            combined_dataset_to_update[encoded_subset_attr_names] = 0
            # set particular feature to one (active)
            combined_dataset_to_update[selected_feature] = 1

            # append updated dataset
            augumented_dataset = augumented_dataset.append(combined_dataset_to_update)

        # augment discrete encodings
        for attribute_name in continuous_set.columns:

            # make a copy of discrete dataset
            combined_dataset_to_update = combined_dataset.copy()
            # set subset of attributes to zero
            combined_dataset_to_update[attribute_name] = combined_dataset_to_update[attribute_name].mean()

            # append updated dataset
            augumented_dataset = augumented_dataset.append(combined_dataset_to_update)

        return augumented_dataset

    # collect distances to the cluster means in the latent space
    def compute_MD(self, z, means):
        def compute_euclid_dist(x, y):
            return np.sqrt(np.sum((x - y) ** 2, axis=1))

        distances = np.apply_along_axis(func1d=compute_euclid_dist, axis=1, arr=z, y=means)
        return np.min(distances, axis=1), np.argmin(distances, axis=1)

    # scale input 1-D array to range [0,1]
    def scale_min_max(self, x_input):
        return (x_input - x_input.min()) / (x_input.max() - x_input.min())