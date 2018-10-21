# Object detection and image processing with AWS Lambda
This example illustrates how object detection and image manipulation can be implemented using Python on AWS Lambda. The code takes an image, uses Tensorflow to detect any humans, and crops the image so that the human is in the middle of the picture.

![Sample Image](https://github.com/iskornienko/find-and-center-with-lambda/blob/master/sample_image.png?raw=true)


### Prepare Lambda Function
1. Go to the AWS Lambda console and create a new Lambda function
2. Set the memory of the Lambda function to 3gig and timeout to 1min
3. Pull in the lambda function code locally or use Cloud9 to edit it (my preference due to the various helper features)
4. Create a directory that will store the various third party libraries and step in to it
```
mkdir vendor
cd vendor
```
5. Pull in version 1.4 of Tensorflow for Object Detection and Pillow for image manipulation
```
pip install --upgrade --ignore-installed --no-cache-dir https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl -t .
touch google/__init__.py
pip install Pillow -t .
```
6. Delete libraries that Tensorflow does not need. This is necessary due to size restrictions imposed on Lambda functions by AWS.
```
rm -r bleach
rm -r concurrent
rm -r external
rm -r html5lib
rm -r markdown
rm -r werkzeug
rm -r tensorboard
find . -name "*.so" | xargs strip
find . -type f -name "*.pyc" -delete
```
7. Copy the object_detection folder in to your project & replace lambda_function.py with the lambda_function.py from this repo
```
cd ..
git clone https://github.com/iskornienko/find-and-center-with-lambda.git
mv find-and-center-with-lambda/vendor/object_detection vendor/
rm lambda_function.py 
mv find-and-center-with-lambda/lambda_function.py .
```

### Prepare S3 Bucket
1. Create a new S3 bucket
2. Create folders input_images, output_images, and model in the bucket
3. Download http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz, unzip it, and upload frozen_inference_graph.pb in to the model folder on s3
4. Go back to the Lambda console and open the function you created
5. In the 'Designer' section, select S3, and scroll down to the Configure trigger section
6. Select the bucket you created in step 1
7. Leave the Event type as object created
8. For the Prefix enter 'input_images'
9. for the Suffic enter 'jpeg'
10. Click Add and then Save in the top right
11. Replace 'mlstuff7631' in lambda_function.py with the name of your S3 bucket
