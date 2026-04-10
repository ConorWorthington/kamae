# Copyright [2024] Expedia, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors
# pylint: disable=no-member
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DataType, DoubleType, FloatType

from kamae.spark.params import (
    MaskValueParams,
    SampleFractionParams,
    SingleInputSingleOutputParams,
)
from kamae.spark.transformers import MinMaxScaleTransformer
from kamae.spark.utils import posexplode_array_for_scaling

from .base import BaseEstimator


class MinMaxScaleEstimator(
    BaseEstimator,
    SampleFractionParams,
    SingleInputSingleOutputParams,
    MaskValueParams,
):
    """
    Min max estimator for use in Spark pipelines.
    This estimator is used to calculate the min and max of the input
    feature column. When fit is called it returns a MinMaxScaleTransformer
    which can be used to standardize/transform additional features.

    WARNING: If the input is an array, we assume that the array has a constant
    shape across all rows.
    """

    @keyword_only
    def __init__(
        self,
        inputCol: Optional[str] = None,
        outputCol: Optional[str] = None,
        inputDtype: Optional[str] = None,
        outputDtype: Optional[str] = None,
        layerName: Optional[str] = None,
        maskValue: Optional[float] = None,
        sampleFraction: Optional[float] = None,
    ) -> None:
        """
        Initializes a MinMaxScaleEstimator estimator.
        Sets all parameters to given inputs.

        :param inputCol: Input column name to standardize.
        :param outputCol: Output column name.
        :param inputDtype: Input data type to cast input column to before
        transforming.
        :param outputDtype: Output data type to cast the output column to after
        transforming.
        :param layerName: Name of the layer. Used as the name of the tensorflow layer
         in the keras model. If not set, we use the uid of the Spark transformer.
        :param maskValue: Value to use for masking. If set, these values will be ignored
        during the computation of the min and max values.
        :param sampleFraction: Fraction of data to sample for statistics
        estimation (exclusive 0.0-1.0). Default None (no sampling).
        :returns: None - class instantiated.
        """
        super().__init__()
        self._setDefault(maskValue=None, sampleFraction=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @property
    def compatible_dtypes(self) -> Optional[List[DataType]]:
        """
        List of compatible data types for the layer.
        If the computation can be performed on any data type, return None.

        :returns: List of compatible data types for the layer.
        """
        return [FloatType(), DoubleType()]

    def _fit(self, dataset: DataFrame) -> "MinMaxScaleTransformer":
        """
        Fits the MinMaxScaleEstimator estimator to the given dataset.
        Calculates the min and max of the input feature column and
        returns a MinMaxScaleTransformer with the min and max set.

        :param dataset: Pyspark dataframe to fit the estimator to.
        :returns: MinMaxScaleTransformer instance with min & max set.
        """
        input_column_type = self.get_column_datatype(dataset, self.getInputCol())
        if not isinstance(input_column_type, ArrayType):
            input_col = F.array(F.col(self.getInputCol()))
            input_column_type = ArrayType(input_column_type)
        else:
            input_col = F.col(self.getInputCol())

        exploded_df = posexplode_array_for_scaling(
            dataset=dataset,
            column=input_col,
            column_datatype=input_column_type,
        )

        if self.getMaskValue() is not None:
            exploded_df = exploded_df.withColumn(
                "val",
                F.when(
                    F.col("val") == F.lit(self.getMaskValue()), F.lit(None)
                ).otherwise(F.col("val")),
            )

        stats_rows = sorted(
            exploded_df.groupBy("pos")
            .agg(
                F.min("val").alias("min"),
                F.max("val").alias("max"),
            )
            .collect(),
            key=lambda row: row["pos"],
        )
        min_vals = [row["min"] for row in stats_rows]
        max_vals = [row["max"] for row in stats_rows]

        return MinMaxScaleTransformer(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            layerName=self.getLayerName(),
            inputDtype=self.getInputDtype(),
            outputDtype=self.getOutputDtype(),
            min=min_vals,
            max=max_vals,
            maskValue=self.getMaskValue(),
        )
