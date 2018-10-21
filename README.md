# find-and-center-with-lambda
Object detection and image processing with AWS Lambda

Prepare Lambda Function
1. Go to the lambda console and create a new lambda function
1. Set the memory to 3gig and timeout to 1min
1. Pull in the lambda function locally or use Cloud9 to edit it (my preferred method)
1. Create a directory that will store all the third party libraries and go in to it
mkdir vendor
cd vendor
1. pull in version 1.4 of tensor flow
pip install --upgrade --ignore-installed --no-cache-dir https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl -t .
1. delete all the unnecessary tensor flow data 
rm -r bleach
rm -r concurrent
rm -r external
rm -r html5lib
rm -r markdown
rm -r werkzeug
rm -r tensorboard
find . -name "*.so" | xargs strip
find . -type f -name "*.pyc" -delete
1. replace lambda_function.py with the lamb_function.py from this github repo
```
cd ..
git clone https://github.com/iskornienko/find-and-center-with-lambda.git
mv find-and-center-with-lambda/vendor/object_detection vendor/
rm lambda_function.py 
mv find-and-center-with-lambda/lambda_function.py .
```
1. copy the object_detection folder in to the vendor directory

Prepare S3 Bucket
1. Create S3 bucket
1. Create folders input_images, output_images, and model
1. Download http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz, unzip it, and upload frozen_inference_graph.pb in to the model folder on s3
1. Go back to the lambda console and open the function you created
1. In the 'designer' section, select S3, and scroll down to the Configure trigger section
1. Select the bucket you created in step 1
1. Leave the Event type as object created
1. For the Prefix enter 'input_images'
1. for the Suffic enter 'jpeg'
1. Click Add and then Save in the top right
