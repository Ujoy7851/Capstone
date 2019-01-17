import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as posenet from '@tensorflow-models/posenet';
import Stats from 'stats.js';
import {drawKeypoints, drawSkeleton, drawBoundingBox} from './demo_util';
//global variables
const stats = new Stats();
let isDetecting;
let stream;
let video;
let canvas;
let ctx;
let knn;
let mobilenet;
//classifiers
const NUM_CLASSES = 7;
const TOPK = 10;

//save& load
let myIncomingClassifier = [];
let myGroups = []

/*
 * 프로그램이 실행되면 실행되는 코드
 */
async function setup(){
    video = await loadVideo();
    await loadCanvas();
    // console.log(video);
    const model = await posenet.load(1.01);
    knn = knnClassifier.create();
    // console.log(knn);
    mobilenet = await mobilenetModule.load();
    // console.log(mobilenet);
    await myloadModel();
    console.log(knn);
    let playAlert = setInterval(async function(){
        if(isDetecting === true){
            // const pose = await 
            let pose = await model.estimateSinglePose(video,0.35,true,16);
            // console.log(pose);
            ctx.clearRect(0,0,640,480);
            if (pose.score >= 0.1) {
                drawKeypoints(pose.keypoints, 0.5, ctx);
                drawSkeleton(pose.keypoints, 0.5, ctx);
            }
            let logits;
            const image = tf.fromPixels(canvas);

            // 'conv_preds' is the logits activation of MobileNet.
            const infer = () => mobilenet.infer(image, 'conv_preds');

            const numClasses = knn.getNumClasses();
            if (numClasses > 0) {

                // If classes have been added run predict
                logits = infer();
                const res = await knn.predictClass(logits, TOPK);
                console.log(res.classIndex + " " + res.confidences[res.classIndex]*100);
                // chrome.tabs.executeScript(null,{code:"scrollBy(0,200);"});
                //control
                if(res.confidences[res.classIndex]*100 > 60){
                    switch(res.classIndex){
                        case 0:
                        break;
                        case 1:
                        scrollBy(0,200);
                        break;
                        case 2:
                        scrollBy(0,-200);
                        break;
                        case 3:
                        break;
                        case 4:
                        break;
                        case 5:
                        break;
                        default:
                    }
                }
            }

            // Dispose image when done
            image.dispose();
            if (logits != null) {
                logits.dispose();
            }
            // console.log(pose);
        }
    },500);
}

/**
 * 카메라가 활성화된 HTMLVideoElement를 생성후 리턴
 */
async function loadVideo(){
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    let video = document.createElement('video');
    video.height = 480;
    video.width = 640;
    video.srcObject = stream;
    video.play();
    return video;
}

async function loadCanvas(){
    canvas = document.createElement('canvas');
    ctx = canvas.getContext('2d');
    // document.body.appendChild(canvas);
    canvas.width = 640;
    canvas.height = 480;
}

async function myloadModel(){
    const myLoadedModel  = await tf.loadModel('https://ujoy7851.github.io/Capstone/model/model.json');
    // console.log('myLoadedModel.layers.length');
    // console.log(myLoadedModel.layers.length);

    const myMaxLayers = myLoadedModel.layers.length;
    const myDenseEnd =  myMaxLayers - 2;
    const myDenseStart = myDenseEnd/2;                                  
    for (let myWeightLoop = myDenseStart; myWeightLoop < myDenseEnd; myWeightLoop++ ){
        // console.log('myLoadedModel.layers['+myWeightLoop+'].getWeights()[0].print(true)');
        myIncomingClassifier[myWeightLoop - myDenseStart] =  myLoadedModel.layers[myWeightLoop].getWeights()[0];
        myGroups[myWeightLoop - myDenseStart] =  myLoadedModel.layers[myWeightLoop].name;                        
    }
    // console.log('Printing all the incoming classifiers');
    for (let x=0;  x < myIncomingClassifier.length; x++){
      myIncomingClassifier[x].print(true);
    }
    // console.log('Activating Classifier');
    knn.dispose()
    knn.setClassifierDataset(myIncomingClassifier);
    // console.log('Classifier loaded');
}

setup();
chrome.runtime.onMessage.addListener(gotMessage);

async function gotMessage(message, sender, sendResponse){
    console.log(message.data);
    if(message.data === "OFF") {
        isDetecting = false;
        // video.pause();
        video.srcObject = null;
        stream.getTracks().forEach((track) => {
            track.stop();
        });
    }
    else if (message.data === "ON"){
        video = await loadVideo();
        isDetecting = true;
    }
}