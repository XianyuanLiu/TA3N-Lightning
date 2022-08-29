# import argparse
import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet, AGKDataSet, EPICDataSet
from utils.loss import *
import pandas as pd

from pathlib import Path
from colorama import init
from colorama import Fore, Back, Style
import numpy as np


def get_train_data_loaders(cfg):
    if cfg.DATASET.MODALITY == 'Audio' or 'RGB' or cfg.DATASET.MODALITY == 'ALL':
        data_length = 1
    elif cfg.DATASET.MODALITY in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    if cfg.DATASET.DATASET == 'epic100':
        train_source_list = Path(cfg.PATHS.TRAIN_SOURCE_LIST)
        train_target_list = Path(cfg.PATHS.TRAIN_TARGET_LIST)
        test_list = Path(cfg.PATHS.TEST_LIST)
    elif cfg.DATASET.DATASET in ['adl']:
        train_source_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.DATASET.DATASET.upper(),
            "annotations/labels_train_test",
            "{}_{}_train.pkl".format(cfg.DATASET.DATASET, cfg.PATHS.DATASET_SOURCE))
        train_target_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.DATASET.DATASET.upper(),
            "annotations/labels_train_test",
            "{}_{}_train.pkl".format(cfg.DATASET.DATASET, cfg.PATHS.DATASET_TARGET))
        test_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.DATASET.DATASET.upper(),
            "annotations/labels_train_test",
            "{}_{}_test.pkl".format(cfg.DATASET.DATASET, cfg.PATHS.DATASET_TARGET))
    elif cfg.DATASET.DATASET in ['epic']:
        train_source_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.DATASET.DATASET.upper(),
            "EPIC_KITCHENS_2018",
            "annotations/labels_train_test",
            "{}_{}_train.pkl".format(cfg.DATASET.DATASET, cfg.PATHS.DATASET_SOURCE))
        train_target_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.DATASET.DATASET.upper(),
            "EPIC_KITCHENS_2018",
            "annotations/labels_train_test",
            "{}_{}_train.pkl".format(cfg.DATASET.DATASET, cfg.PATHS.DATASET_TARGET))
        test_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.DATASET.DATASET.upper(),
            "EPIC_KITCHENS_2018",
            "annotations/labels_train_test",
            "{}_{}_test.pkl".format(cfg.DATASET.DATASET, cfg.PATHS.DATASET_TARGET))
    else:
        train_source_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.PATHS.DATASET_SOURCE.upper(),
            "annotations/labels_train_test",
            "{}_train.pkl".format(cfg.PATHS.DATASET_SOURCE))
        train_target_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.PATHS.DATASET_TARGET.upper(),
            "annotations/labels_train_test",
            "{}_train.pkl".format(cfg.PATHS.DATASET_TARGET))
        test_list = Path(cfg.PATHS.PATH_DATA_ROOT).joinpath(
            cfg.PATHS.DATASET_TARGET.upper(),
            "annotations/labels_train_test",
            "{}_test.pkl".format(cfg.PATHS.DATASET_TARGET))

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    # num_source = len(pd.read_pickle(cfg.PATHS.TRAIN_SOURCE_LIST).index)
    # num_target = len(pd.read_pickle(cfg.PATHS.TRAIN_TARGET_LIST).index)
    # num_val = len(pd.read_pickle(cfg.PATHS.TEST_LIST).index)

    num_source = len(pd.read_pickle(train_source_list).index)
    num_target = len(pd.read_pickle(train_target_list).index)
    num_val = len(pd.read_pickle(test_list).index)

    num_iter_source = num_source / cfg.TRAINER.BATCH_SIZE[0]
    num_iter_target = num_target / cfg.TRAINER.BATCH_SIZE[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[
                                                                              0] == 'Y' else num_source
    num_target_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[
                                                                              1] == 'Y' else num_target
    num_source_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[
                                                                            0] == 'Y' else num_source
    num_target_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[
                                                                            1] == 'Y' else num_target
    if cfg.DATASET.DATASET == 'epic100':
        train_source_data = Path(cfg.PATHS.PATH_DATA_SOURCE + ".pkl")
        source_set = TSNDataSet(train_source_data, train_source_list,
                                num_dataload=num_source_train,
                                num_segments=cfg.DATASET.NUM_SEGMENTS,
                                new_length=data_length, modality=cfg.DATASET.MODALITY,
                                image_tmpl="none",
                                random_shift=False,
                                test_mode=True,
                                )
    else:
        train_source_data = Path(cfg.PATHS.PATH_DATA_SOURCE)
        if cfg.DATASET.DATASET in ['adl', 'gtea', 'kitchen']:
            source_set = AGKDataSet(train_source_data, train_source_list,
                                    num_dataload=num_source_train,
                                    num_segments=cfg.DATASET.NUM_SEGMENTS,
                                    new_length=data_length, modality=cfg.DATASET.MODALITY,
                                    image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                           "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                    random_shift=False,
                                    test_mode=True,
                                    )
        elif cfg.DATASET.DATASET == 'epic':
            source_set = EPICDataSet(train_source_data.joinpath("train"), train_source_list,
                                     num_dataload=num_source_train,
                                     num_segments=cfg.DATASET.NUM_SEGMENTS,
                                     new_length=data_length, modality=cfg.DATASET.MODALITY,
                                     image_tmpl="img_{:07d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff",
                                                                                            "RGBDiff2",
                                                                                            "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                     random_shift=False,
                                     test_mode=True,
                                     )
        else:
            ValueError("Wrong dataset option.")

    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=cfg.TRAINER.BATCH_SIZE[0], shuffle=False,
                                                sampler=source_sampler, num_workers=cfg.TRAINER.WORKERS,
                                                pin_memory=True)

    if cfg.DATASET.DATASET == 'epic100':
        train_target_data = Path(cfg.PATHS.PATH_DATA_TARGET + ".pkl")
        target_set = TSNDataSet(train_target_data, train_target_list,
                                num_dataload=num_target_train,
                                num_segments=cfg.DATASET.NUM_SEGMENTS,
                                new_length=data_length, modality=cfg.DATASET.MODALITY,
                                image_tmpl="none",
                                random_shift=False,
                                test_mode=True,
                                )
    else:
        train_target_data = Path(cfg.PATHS.PATH_DATA_TARGET)
        if cfg.DATASET.DATASET in ['adl', 'gtea', 'kitchen']:
            target_set = AGKDataSet(train_target_data, train_target_list,
                                    num_dataload=num_target_train,
                                    num_segments=cfg.DATASET.NUM_SEGMENTS,
                                    new_length=data_length, modality=cfg.DATASET.MODALITY,
                                    image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff", "RGBDiff2",
                                                                                           "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                    random_shift=False,
                                    test_mode=True,
                                    )
        elif cfg.DATASET.DATASET == 'epic':
            target_set = EPICDataSet(train_target_data.joinpath("train"), train_target_list,
                                     num_dataload=num_target_train,
                                     num_segments=cfg.DATASET.NUM_SEGMENTS,
                                     new_length=data_length, modality=cfg.DATASET.MODALITY,
                                     image_tmpl="img_{:07d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff",
                                                                                            "RGBDiff2",
                                                                                            "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                     random_shift=False,
                                     test_mode=True,
                                     )

    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=cfg.TRAINER.BATCH_SIZE[1], shuffle=False,
                                                sampler=target_sampler, num_workers=cfg.TRAINER.WORKERS,
                                                pin_memory=True)

    return source_loader, target_loader


def get_val_data_loaders(cfg):
    if cfg.DATASET.MODALITY == 'Audio' or 'RGB' or cfg.DATASET.MODALITY == 'ALL':
        data_length = 1
    elif cfg.DATASET.MODALITY in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
        data_length = 1

    # calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
    num_source = len(pd.read_pickle(cfg.PATHS.TRAIN_SOURCE_LIST).index)
    num_target = len(pd.read_pickle(cfg.PATHS.TRAIN_TARGET_LIST).index)
    num_val = len(pd.read_pickle(cfg.PATHS.TEST_LIST).index)

    num_iter_source = num_source / cfg.TRAINER.BATCH_SIZE[0]
    num_iter_target = num_target / cfg.TRAINER.BATCH_SIZE[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[
                                                                              0] == 'Y' else num_source
    num_target_train = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[
                                                                              1] == 'Y' else num_target
    num_source_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[0]) if cfg.TRAINER.COPY_LIST[0] == 'Y' else num_source
    num_target_val = round(num_max_iter * cfg.TRAINER.BATCH_SIZE[1]) if cfg.TRAINER.COPY_LIST[1] == 'Y' else num_target

    val_source_list = Path(cfg.PATHS.VAL_SOURCE_LIST)
    if cfg.DATASET.DATASET == 'epic100':
        val_source_data = Path(cfg.PATHS.PATH_VAL_DATA_SOURCE + ".pkl")
        source_set_val = TSNDataSet(val_source_data, val_source_list,
                                    num_dataload=num_source_val,
                                    num_segments=cfg.DATASET.VAL_SEGMENTS,
                                    new_length=data_length, modality=cfg.DATASET.MODALITY,
                                    image_tmpl="none",
                                    random_shift=False,
                                    test_mode=True,
                                    )
    else:
        val_source_data = Path(cfg.PATHS.PATH_VAL_DATA_SOURCE)
        if cfg.DATASET.DATASET in ['adl', 'gtea', 'kitchen']:
            source_set_val = AGKDataSet(val_source_data, val_source_list,
                                        num_dataload=num_source_val,
                                        num_segments=cfg.DATASET.VAL_SEGMENTS,
                                        new_length=data_length, modality=cfg.DATASET.MODALITY,
                                        image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff",
                                                                                               "RGBDiff2",
                                                                                               "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                        random_shift=False,
                                        test_mode=True,
                                        )
        elif cfg.DATASET.DATASET == 'epic':
            source_set_val = EPICDataSet(val_source_data.joinpath("train"), val_source_list,
                                         num_dataload=num_source_val,
                                         num_segments=cfg.DATASET.VAL_SEGMENTS,
                                         new_length=data_length, modality=cfg.DATASET.MODALITY,
                                         image_tmpl="img_{:07d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff",
                                                                                                "RGBDiff2",
                                                                                                "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                         random_shift=False,
                                         test_mode=True,
                                         )

    # source_sampler_val = torch.utils.data.sampler.RandomSampler(source_set_val)
    source_loader_val = torch.utils.data.DataLoader(source_set_val, batch_size=cfg.TRAINER.BATCH_SIZE[0], shuffle=False,
                                                    num_workers=cfg.TRAINER.WORKERS, pin_memory=True)

    val_target_list = Path(cfg.PATHS.VAL_TARGET_LIST)
    if cfg.DATASET.DATASET == 'epic100':
        val_target_data = Path(cfg.PATHS.PATH_VAL_DATA_TARGET + ".pkl")
        target_set_val = TSNDataSet(val_target_data, val_target_list,
                                    num_dataload=num_target_val,
                                    num_segments=cfg.DATASET.VAL_SEGMENTS,
                                    new_length=data_length, modality=cfg.DATASET.MODALITY,
                                    image_tmpl="none",
                                    random_shift=False,
                                    test_mode=True,
                                    )
    else:
        val_target_data = Path(cfg.PATHS.PATH_VAL_DATA_TARGET)
        if cfg.DATASET.DATASET in ['adl', 'gtea', 'kitchen']:
            target_set_val = AGKDataSet(val_target_data, val_target_list,
                                        num_dataload=num_target_val,
                                        num_segments=cfg.DATASET.VAL_SEGMENTS,
                                        new_length=data_length, modality=cfg.DATASET.MODALITY,
                                        image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff",
                                                                                               "RGBDiff2",
                                                                                               "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                        random_shift=False,
                                        test_mode=True,
                                        )
        elif cfg.DATASET.DATASET == 'epic':
            target_set_val = EPICDataSet(val_target_data.joinpath("test"), val_target_list,
                                         num_dataload=num_target_val,
                                         num_segments=cfg.DATASET.VAL_SEGMENTS,
                                         new_length=data_length, modality=cfg.DATASET.MODALITY,
                                         image_tmpl="img_{:07d}.t7" if cfg.DATASET.MODALITY in ["RGB", "RGBDiff",
                                                                                                "RGBDiff2",
                                                                                                "RGBDiffplus"] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                         random_shift=False,
                                         test_mode=True,
                                         )

    # target_sampler_val = torch.utils.data.sampler.RandomSampler(target_set_val)
    target_loader_val = torch.utils.data.DataLoader(target_set_val, batch_size=cfg.TRAINER.BATCH_SIZE[1], shuffle=False,
                                                    num_workers=cfg.TRAINER.WORKERS, pin_memory=True)

    return source_loader_val, target_loader_val


def get_test_data_loaders(cfg):
    data_length = 1 if cfg.DATASET.MODALITY == "RGB" else 1
    num_test = len(pd.read_pickle(cfg.PATHS.TEST_LIST).index)

    test_target_list = Path(cfg.PATHS.TEST_LIST)
    if cfg.DATASET.DATASET == 'epic100':
        test_target_data = Path(cfg.TESTER.TEST_TARGET_DATA + ".pkl")
        if cfg.TESTER.NOUN_TARGET_DATA is not None:
            data_set = TSNDataSet(test_target_data, test_target_list,
                                  num_dataload=num_test,
                                  num_segments=cfg.TESTER.TEST_SEGMENTS,
                                  new_length=data_length, modality=cfg.DATASET.MODALITY,
                                  image_tmpl="none",
                                  test_mode=True, noun_data_path=cfg.TESTER.NOUN_TARGET_DATA + ".pkl"
                                  )
        else:
            data_set = TSNDataSet(test_target_data, test_target_list,
                                  num_dataload=num_test,
                                  num_segments=cfg.TESTER.TEST_SEGMENTS,
                                  new_length=data_length, modality=cfg.DATASET.MODALITY,
                                  image_tmpl="none",
                                  test_mode=True
                                  )
    else:
        test_target_data = Path(cfg.TESTER.TEST_TARGET_DATA)
        if cfg.DATASET.DATASET in ['adl', 'gtea', 'kitchen']:
            if cfg.TESTER.NOUN_TARGET_DATA is not None:
                data_set = AGKDataSet(test_target_data, test_target_list,
                                      num_dataload=num_test,
                                      num_segments=cfg.TESTER.TEST_SEGMENTS,
                                      new_length=data_length, modality=cfg.DATASET.MODALITY,
                                      image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ['RGB', 'RGBDiff',
                                                                                             'RGBDiff2',
                                                                                             'RGBDiffplus'] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                      test_mode=True, noun_data_path=cfg.TESTER.NOUN_TARGET_DATA + ".pkl"
                                      )
            else:
                data_set = AGKDataSet(test_target_data, test_target_list,
                                      num_dataload=num_test,
                                      num_segments=cfg.TESTER.TEST_SEGMENTS,
                                      new_length=data_length, modality=cfg.DATASET.MODALITY,
                                      image_tmpl="img_{:05d}.t7" if cfg.DATASET.MODALITY in ['RGB', 'RGBDiff',
                                                                                             'RGBDiff2',
                                                                                             'RGBDiffplus'] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                      test_mode=True
                                      )
        elif cfg.DATASET.DATASET == 'epic':
            if cfg.TESTER.NOUN_TARGET_DATA is not None:
                data_set = EPICDataSet(test_target_data.joinpath("test"), test_target_list,
                                       num_dataload=num_test,
                                       num_segments=cfg.TESTER.TEST_SEGMENTS,
                                       new_length=data_length, modality=cfg.DATASET.MODALITY,
                                       image_tmpl="img_{:07d}.t7" if cfg.DATASET.MODALITY in ['RGB', 'RGBDiff',
                                                                                              'RGBDiff2',
                                                                                              'RGBDiffplus'] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                       test_mode=True, noun_data_path=cfg.TESTER.NOUN_TARGET_DATA + ".pkl"
                                       )
            else:
                data_set = EPICDataSet(test_target_data.joinpath("test"), test_target_list,
                                       num_dataload=num_test,
                                       num_segments=cfg.TESTER.TEST_SEGMENTS,
                                       new_length=data_length, modality=cfg.DATASET.MODALITY,
                                       image_tmpl="img_{:07d}.t7" if cfg.DATASET.MODALITY in ['RGB', 'RGBDiff',
                                                                                              'RGBDiff2',
                                                                                              'RGBDiffplus'] else cfg.MODEL.FLOW_PREFIX + "{}_{:05d}.t7",
                                       test_mode=True
                                       )
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=cfg.TESTER.BATCH_SIZE, shuffle=False,
                                              num_workers=cfg.TRAINER.WORKERS, pin_memory=True)

    return data_loader
