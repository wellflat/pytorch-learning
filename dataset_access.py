#!/usr/bin/env python
# -*- coding: utf-8 -*-

from allegroai import DatasetVersion, DataView, SingleFrame, Task
from time import sleep

if __name__ == '__main__':
    task = Task.init(
        project_name="sample project",
        task_name="register buckets"
    )
    # Get the version we want to update
    dataset_name = 'Data registration example2'
    #version_name = 'NSFW,Jan'
    version_name = 'Gun,Jan'
    try:
        version = DatasetVersion.get_version(
            dataset_name=dataset_name,
            version_name=version_name
        )
        #print(version)     

        dataview = DataView()
        #roi_query='good'
        #roi_query='no_good'
        #roi_query='modelgun_good'
        roi_query='SM'
        dataview.add_query(
            dataset_name=dataset_name,
            version_name=version_name,
            roi_query=roi_query
        )
        c = 0
        for frame in dataview.get_iterator():
            #print(frame)
            #local_file = frame.get_local_source()
            #print(local_file)
            annotation = frame.get_annotations()
            print(annotation)
            c += 1
        
        print(c)
        
    except ValueError:
        print(f"Can not find version {version_name}, will create a new one in {dataset_name} dataset")
        version = DatasetVersion.create_version(dataset_name=dataset_name, version_name=version_name)
    