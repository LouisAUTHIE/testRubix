<?php

namespace Chap5\Analyse;

ini_set('memory_limit', '-1');
ini_set('max_execution_time', 20000);
require_once __DIR__ . '../../vendor/autoload.php';

use function Rubix\ML\iterator_contains_nan;

use Php2plotly\basic\PieChart;
use Php2plotly\basic\ScatterMapBox;
use Php2plotly\basic\ScatterPlot;
use Php2plotly\preprocessor\Preprocessor;
use Php2plotly\scientific\Heatmap;
use Php2plotly\stats\BoxPlot;
use Php2plotly\stats\Histogram;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Kernels\SVM\Linear;
use Rubix\ML\Kernels\SVM\Polynomial;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Regressors\SVR;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

?>

<html>
    <head>

        <title>Regression</title>
        <script src="../../assets/js/plotly-2.32.0.min.js" charset="utf-8"></script>
    </head>
    <body>
        <h1>Regression</h1>

<?php
//Extract Dataset
$columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'];
$extractorHousingCSV = new CSV('housing.csv', true, ',', '"');

$samplesContinuous = new ColumnPicker($extractorHousingCSV, ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']);
$sampleCategorical = new ColumnPicker($extractorHousingCSV, ['ocean_proximity']);

//Process categorical data
$proximityDataset = Unlabeled::fromIterator($sampleCategorical);
$oneHotEncoder = new OneHotEncoder();
$proximityDataset->apply($oneHotEncoder);

//Process continuous data
$datasetContinuous = Labeled::fromIterator($samplesContinuous);

$numericTransformer = new NumericStringConverter();
$datasetContinuous->apply($numericTransformer);

//Regroup the datasets
//$dataset = $datasetContinuous->join($proximityDataset);

$dataset = $datasetContinuous;

//Remove duplocates
$dataset->deduplicate();


//Check for missing values
$missingValueCategorical = '?';
$missingValues = function ($record) use ($missingValueCategorical){
    $missesContinuousValue = iterator_contains_nan($record);
    $missesCategoricalValue = in_array($missingValueCategorical, $record);
    return $missesCategoricalValue || $missesContinuousValue;
};

$incompleteRecords = $dataset->filter($missingValues);
$numberOfIncompleteRecords = count($incompleteRecords);

echo "Number of incomplete records: $numberOfIncompleteRecords";

$missingValueCategorical = '?';
$cleanRecord = function ($record) use ($missingValueCategorical){
    $missesContinuousValue = iterator_contains_nan($record);
    $missesCategoricalValue = in_array($missingValueCategorical, $record);
    return !$missesCategoricalValue && !$missesContinuousValue && $record[2] < 52 && $record[3] < 5811 && $record[4] < 1196 && $record[5] < 3205 && $record[6] < 1108 && $record[7] < 8;
};
$dataset = $dataset->filter($cleanRecord);


//Scale continuous features for SVR
$dataSetToScaleContinuous = new Unlabeled($dataset->samples());
$dataSetToScaleContinuous = $dataSetToScaleContinuous->join(Unlabeled::fromIterator($dataset->labels()));
$dataSetToScaleContinuous->apply($numericTransformer);

//Remove outliers for labels
$maxLabelValue = 500000;
$belowLimitRecord = function ($record) use ($maxLabelValue){
    return $record[8] < $maxLabelValue;
};
//$dataSetToScaleContinuous = $dataSetToScaleContinuous->filter($belowLimitRecord);


$minMaxScaler = new MinMaxNormalizer();
$zcScaler = new ZScaleStandardizer();
$dataSetToScaleContinuous->apply($zcScaler);

$labelsScaled = $dataSetToScaleContinuous->feature(8);
$dataSetToScaleContinuous->dropFeature(8);

$datasetScaled = new Labeled($dataSetToScaleContinuous->samples(), $labelsScaled);

//Data visualization
//Get continuous features
$types = $dataset->types();
$continuousFeatures = array_keys(array_filter($types, function ($type){
    return $type->code() == DataType::CONTINUOUS;
}));

//Display continuous features into plotly graph
$data = [];
$histograms = [];
echo '<div style="display:flex; flex-wrap: wrap;">';
foreach($continuousFeatures as $key=>$feature){
    if($key >= 8){
        continue;
    }
    $data = $dataset->feature($feature);
    $histograms[] = new Histogram('continuous_'.$key, ['x' => $data]);
    echo '<div style="text-align:center;"><div id="continuous_'.$key.'" style="width:600px;height:400px;"></div>'.$columns[$key].'</div>';
    echo '<script>'.$histograms[$key]->render().'</script>';
}

$dataLabels = $dataset->labels();
$histoLabels = new Histogram('continuous_labels', ['x' => $dataLabels]);
echo '<div style="text-align:center;"><div id="continuous_labels" style="width:600px;height:400px;"></div>Labels</div>';
echo '<script>'.$histoLabels->render().'</script>';


//Display box plots to see if there are outliers
echo '<div style="text-align:center;"><div id="bplot" style="width:600px;height:400px;"></div>Box plots</div>';
$listsBoxPlot = [];
foreach($continuousFeatures as $key=>$feature){
    if($key >= 8){
        continue;
    }
    $data = $dataset->feature($feature);
    $listsBoxPlot[] = ['y' => $data];
}
$boxplot = new BoxPlot('bplot', $listsBoxPlot);
echo '<script>'.$boxplot->render().'</script>';

//Correlation matrix
$correlationMatrix = Preprocessor::buildArrayForPearsonCollerationMatrix(array_merge($dataset->features(), [$dataset->labels()]));
echo '<div style="text-align:center;"><div id="heatmap" style="width:600px;height:400px;"></div>Correlation Heatmap</div>';
$heatmap = new Heatmap('heatmap', ["x"=>$continuousFeatures, "y"=>$continuousFeatures, "z"=>$correlationMatrix]);
echo '<script>'.$heatmap->render().'</script>';


echo '<div id="mapbox" style="width:600px;height:400px;"></div>';
$lat = [];
$lon = [];
$dataGeo = [];
$colors = [];
foreach($dataset->samples() as $key=>$sample){
    $color = "blue";
    if ($dataset->label($key) < 200000 ){
         $colors[0] = "rgba(0, 0, 255, 0.8)";
         $lat[0][] = $sample[1];
         $lon[0][] = $sample[0];
    }else if($dataset->label($key) < 300000){
        $colors[1] = "rgba(0, 255, 50, 0.8)";
        $lat[1][] = $sample[1];
        $lon[1][] = $sample[0];
    }else{
        $colors[2] = "rgba(255, 0, 0, 0.8)";
        $lat[2][] = $sample[1];
        $lon[2][] = $sample[0];
    }
}

foreach($colors as $key=>$color)
    $dataGeo[] = ['lon' => $lon[$key], 'lat' => $lat[$key], 'text'=>[], 'marker'=> ['color' => $color, 'size' => 5]];

$layout = ['height' => 400, 'width' => 600, 'centerlat' => 37, 'centerlon' => -122, 'zoom' => 4];
$scatterMapBox = new ScatterMapBox('mapbox', $dataGeo, $layout);
echo '<script>'.$scatterMapBox->render().'</script>';

echo '</div>';

$dataset->randomize();
$datasetScaled->randomize();
$dataset->transformLabels('floatval');
$datasetScaled->transformLabels('floatval');

//Split test and training sets
[$testing, $training] = $dataset->split(0.2);
[$testingScaled, $trainingScaled] = $datasetScaled->split(0.2);

$estimatorRidge = new Ridge();
$estimatorSVR = new SVR();

$RSquared = new RSquared();

$kfold = new KFold(10);
$scoreSVR = $kfold->test($estimatorSVR, $trainingScaled, $RSquared);
$scoreRidge = $kfold->test($estimatorRidge, $training, $RSquared);

echo "<h1>Regression Analysis</h1>";
echo "<h2><u>RSquared Score</u></h2>";
echo "<br>";
echo "Ridge: ";
echo $scoreRidge;

echo "<br>";
echo "SVR : ";
echo $scoreSVR;



$testingPredictionsRidge = $estimatorRidge->predict($testing);
$testingPredictionsSVR = $estimatorSVR->predict($testingScaled);
$scoreRSquaredRidge = $RSquared->score($testingPredictionsRidge, $testing->labels());
$scoreRSquaredSVR = $RSquared->score($testingPredictionsSVR, $testingScaled->labels());



echo "<h2><u>RSquared Score on Testing Set</u></h2>";
echo "<br>";
echo "Ridge: ";
echo $scoreRSquaredRidge;

echo "<br>";
echo "SVR : ";
echo $scoreRSquaredSVR;

echo "<br>";



?>
 



