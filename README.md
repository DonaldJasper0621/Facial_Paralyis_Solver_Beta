# facial_paralysis_solver

# Solution


1. Transformation of models with OpenVINO Optimizer
2. Test environment with OpenVINO Benchmark
3. Mobile device camera to acquire frontal photos of users
4. Ganimation model: Converts photos into 16 photos with different face action units
5. face_mesh model: capture the position of contours and organs on the face and target expression photos
6. landmark_accuracy algorithm: compare the difference between the target expression and the position of the user's organs in the camera, and compare the two for similarity

![螢幕擷取畫面-2022-12-11-231824](https://user-images.githubusercontent.com/74034793/208851206-469da6e5-3134-463c-8201-2df8595528cf.jpg)

Facial Paralysis Solver is divided into three parts.

According to the research results of psychologist Paul Ekman, human facial expressions can be categorized into more than 40 different combinations of facial action units (AUs). The team collected 100 photos of paralyzed patients in advance, identified the AUs with paralysis characteristics through Openface, and entered them into Ganimation together with the user's photos to generate 16 new photos with different expressions. Most of the resulting expressions showed common rehabilitation actions such as laughing and frowning.

In the second stage, after obtaining the replacement photos with different expressions, we expect users to train their faces with the rehabilitated expressions in the photos as the target.

In the final stage, we convert the difference between the user's real expression and the restored expression into a component based on the relative position and size of the face organs, indicating the completion of the user's training. At the same time, the system will also prompt the user to make adjustments such as raising the eyebrows according to the position of the five senses. Finally, the system will present the overall performance of the training to the user.

The above two models, Ganimation and Facemesh, both use the Model Optimizer provided by OpenVINO to convert the onnx model into IR model for subsequent optimization of the terminal machine.

# Team Introduction
We are a group from the Department of Information Management of National Chengchi University, led by Professor Fang Yu.

### Prerequisite
- NoodJs 16
- yarn


## Backend 

### Prerequisite
- Python 3.9
- Poetry
- Pre-commit


## Common-used Git 
Get the latest version of code in the remote main branch
```cmd
git pull origin main
```
Stage the update that you currently made

```cmd
git add --a
```
Commit the latest change to a log
```cmd
git commit -m '<your message>'
```

Push the latest version of code in the local main branch
```cmd
git push origin main
```
How to start: npm run dev
```
