// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as posenet from '@tensorflow-models/posenet';
import {drawKeypoints, drawSkeleton, drawBoundingBox} from './demo_util';
import Stats from 'stats.js';

// Number of classes to classify
const NUM_CLASSES = 4;
// Webcam Image size. Must be 227. 
//const IMAGE_SIZE = 227;
const videoWidth = 600;
const videoHeight = 500;

// K value for KNN
const TOPK = 10;
const stats = new Stats();

let myIncomingClassifier = [];
let myGroups = []

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');
    this.video.setAttribute('hidden', true);

    // Add video element to DOM
    document.body.appendChild(this.video);

    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    document.body.appendChild(this.canvas);

    this.canvas.width = videoWidth;
    this.canvas.height = videoHeight;

    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Pose " + i;
      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);

      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }

    this.sbutton = document.createElement('button')
    this.sbutton.innerText = "Save model";
    document.body.appendChild(this.sbutton);

    this.sbutton.addEventListener('click', () => {this.mysaveModel()});

    this.lbutton = document.createElement('button')
    this.lbutton.innerText = "load model";
    document.body.appendChild(this.lbutton);

    this.lbutton.addEventListener('click', () => {this.myloadModel()});

    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = videoWidth;
        this.video.height = videoHeight;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.net = await posenet.load(1.0);
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    //this.myloadModel();
        
    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {

      let minPoseConfidence = 0.1;
      let minPartConfidence = 0.5;
      const pose = await this.net.estimateSinglePose(
      this.video, 0.5, true, 16);
  
      this.ctx.clearRect(0, 0, videoWidth, videoHeight);
  
      this.ctx.save();
      this.ctx.scale(-1, 1);
      this.ctx.translate(-videoWidth, 0);
      //this.ctx.drawImage(this.video, 0, 0, videoWidth, videoHeight);
      this.ctx.restore();
  
      // For each pose (i.e. person) detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores

        if (pose.score >= minPoseConfidence) {
          drawKeypoints(pose.keypoints, minPartConfidence, this.ctx);
          drawSkeleton(pose.keypoints, minPartConfidence, this.ctx);
        }

      // Get image data from video element
      const image = tf.fromPixels(this.canvas);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  //save and load model
  async defineClassifierModel(myPassedClassifier){
    let myLayerList = [];
     myLayerList[0] = [];    // for the input layer name as a string
     myLayerList[1] = [];    // for the input layer
     myLayerList[2] = [];    // for the concatenate layer name as a string
     myLayerList[3] = [];    // for the concatenate layer
                                                                                                                                                                                        
    let myMaxClasses = myPassedClassifier.getNumClasses();
                                                                                             
    for (let myClassifierLoop = 0; myClassifierLoop < myMaxClasses; myClassifierLoop++){
      myLayerList[0][myClassifierLoop] = 'myInput'  + myClassifierLoop;
      console.log('define input for'+ myClassifierLoop);
      myLayerList[1][myClassifierLoop] = tf.input({shape: myPassedClassifier.getClassifierDataset()[myClassifierLoop].shape[0], name: myLayerList[1][myClassifierLoop]});
      console.log('define dense for: '+myClassifierLoop);
      myLayerList[2][myClassifierLoop] = 'myInput'+myClassifierLoop+'Dense1';
      myLayerList[3][myClassifierLoop] = tf.layers.dense({units: 1000, name: myGroups[myClassifierLoop]}).apply(myLayerList[1][myClassifierLoop]);
    }
                                                                                             
   console.log('Concatenate Paths');
   const myConcatenate1 = tf.layers.concatenate({axis : 1, name: 'myConcatenate1'}).apply(myLayerList[3]);
   const myConcatenate1Dense4 = tf.layers.dense({units: 1, name: 'myConcatenate1Dense4'}).apply(myConcatenate1);
   console.log('Define Model');

   const myClassifierModel = tf.model({inputs: myLayerList[1], outputs: myConcatenate1Dense4});                                                         
   myClassifierModel.summary();
   console.log('myClassifierModel.layers[myMaxClasses]');
   console.log(myClassifierModel.layers[myMaxClasses]);
   myPassedClassifier.getClassifierDataset()[0].print(true);

   for (let myClassifierLoop = 0; myClassifierLoop < myMaxClasses; myClassifierLoop++ ){
     const myInWeight = await myPassedClassifier.getClassifierDataset()[myClassifierLoop];
     myClassifierModel.layers[myClassifierLoop + myMaxClasses].setWeights([myInWeight, tf.ones([1000])]);
    }
    return  myClassifierModel;
  }

  async mysaveModel(){
    const myClassifierModel2 = await this.defineClassifierModel(this.knn);
    myClassifierModel2.save('downloads://classifiermodel');                                                                       
  }

  async myloadModel(){
    const myLoadedModel  = await tf.loadModel('https://localhost:9966/model.json');
    console.log('myLoadedModel.layers.length');
    console.log(myLoadedModel.layers.length);

    const myMaxLayers = myLoadedModel.layers.length;
    const myDenseEnd =  myMaxLayers - 2;
    const myDenseStart = myDenseEnd/2;                                  
    for (let myWeightLoop = myDenseStart; myWeightLoop < myDenseEnd; myWeightLoop++ ){
        console.log('myLoadedModel.layers['+myWeightLoop+'].getWeights()[0].print(true)');
        myIncomingClassifier[myWeightLoop - myDenseStart] =  myLoadedModel.layers[myWeightLoop].getWeights()[0];
        myGroups[myWeightLoop - myDenseStart] =  myLoadedModel.layers[myWeightLoop].name;                        
    }
    console.log('Printing all the incoming classifiers');
    for (x=0;  x < myIncomingClassifier.length ; x++){
      myIncomingClassifier[x].print(true);
    }
    console.log('Activating Classifier');
    this.knn.dispose()
    this.knn.setClassifierDataset(myIncomingClassifier);
    console.log('Classifier loaded');
  }
}

window.addEventListener('load', () => new Main());