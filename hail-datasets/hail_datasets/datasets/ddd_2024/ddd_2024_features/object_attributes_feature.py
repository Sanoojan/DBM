import copy
import numpy as np

from ddd_2024_features.base_feature import BaseFeature, BaseCollatedFeature 

   
# Object attributes - data constant over time, but variable in number of objects
class ObjectAttributesFeature(BaseFeature):
    def __init__(self, feature_name, feature_attributes, full_scenario_path, sample):
        super().__init__(feature_name, feature_attributes)
        
        # Load CSV data
        csv_path = self.get_feature_path(full_scenario_path, feature_attributes["file_name"] + ".csv")
        if "column_types" in feature_attributes:
            feature_attributes["columns"] = list(feature_attributes["column_types"].keys())
        csv_columns = feature_attributes["columns"] if "columns" in feature_attributes else None
        rename_dict = {feature_attributes["object_id_column"]: "object_id"} if "object_id_column" in feature_attributes else None
        self.attributes = self.load_dataframe(csv_path, index="object_id", column_list=csv_columns, rename_dict=rename_dict)
            
        # Only include object attributes which intersect with the chunk time window if start_epoch_ns and end_epoch_ns included
        self.attributes = sample.chunk.filter_intersections(self.attributes)
        if not self.cache_features:
            self.attributes = self.attributes.copy()

class CollatedObjectAttributesFeature(BaseCollatedFeature):
    # Populates the following:
    # self.attributes Dictionary from attribute_name to (batch_size, max_objects)
    # self.attributes_valid Dictionary from attribute_name to (batch_size, max_objects)
    # self.attributes_columns
    
    class AttributeData():
        def __init__(self, batch_size, max_objects, column_type):
            self.data = np.zeros((batch_size, max_objects), column_type)
            self.valid = np.zeros((batch_size, max_objects))
    
    def __init__(self, feature_name, feature_attributes, list_of_features, collated_sample):
        super().__init__(feature_name, feature_attributes)
                
        batch_size = collated_sample.batch_size
        if "max_objects" not in feature_attributes:
            raise Exception(f"Must declare max objects for feature {feature_name} when using collation.")
        max_objects = feature_attributes["max_objects"]

        # Ensure object_id is always included
        column_types = copy.deepcopy(feature_attributes["column_types"])
        column_types["object_id"] = np.int64

        # Initialize outputs
        self.attributes = {}
        self.attributes_valid = {}
        self.attributes_columns = list(column_types.keys())
        for column_name, column_type in column_types.items():
            self.attributes[column_name] = np.zeros((batch_size, max_objects), column_type)
            self.attributes_valid[column_name] = np.zeros((batch_size, max_objects))

        # Loop through samples and add the attributes
        for sample_on, feature_data in enumerate(list_of_features):
            if feature_data is not None:
                # Add all objects for the sample
                attribute_data = feature_data.attributes
                attribute_data.reset_index(inplace=True)
                objects_used = min(attribute_data.shape[0], max_objects)
                for column_name in column_types.keys():
                    if column_name in attribute_data.columns:
                        self.attributes_valid[column_name][sample_on, :objects_used] = 1
                        self.attributes[column_name][sample_on, :objects_used] = attribute_data[
                            column_name
                        ].iloc[:objects_used]
