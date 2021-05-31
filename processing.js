var model;

async function loadModel() {
    model = await tf.loadGraphModel("TFJS/model.json");
}

function predictImage(){
    //    console.log("Processing...");
    let img = cv.imread(canvas);
    cv.cvtColor(img, img, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(img, img, 175, 255, cv.THRESH_BINARY);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(img, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    let rect = cv.boundingRect(contours.get(0));
    img = img.roi(rect);

    var height = img.rows;
    var width = img.cols;

    if (height > width) {var scaleFactor = 20 / height;}
    else {var scaleFactor = 20 / width; }
    cv.resize(img, img, new cv.Size(0, 0), scaleFactor, scaleFactor, cv.INTER_AREA);

    height = img.rows;
    width = img.cols;

    const  LEFT = Math.ceil(4 + (20 - width) / 2);
    const  RIGHT = Math.floor(4 + (20 - width) / 2);
    const  TOP = Math.ceil(4 + (20 - height) / 2);
    const  BOTTOM = Math.floor(4 + (20 - height) / 2);

    cv.copyMakeBorder(img, img, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    //    console.log(`TOP ${TOP}, BOTTOM ${BOTTOM}, LEFT ${LEFT}, RIGHT ${RIGHT}`);

    // Center of Mass
    cv.findContours(img, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);
    const cx = Moments.m10 / Moments.m00;
    const cy = Moments.m01 / Moments.m00;

    //    console.log(`MOO: ${Moments.m00} | cx: ${cx} | cy: ${cy}`);
    const X_SHIFT = Math.round(img.cols/2.0 - cx);
    const Y_SHIFT = Math.round(img.rows/2.0 - cy);

    let M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    cv.warpAffine(img, img, M, new cv.Size(img.rows, img.cols), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

    let pxValues = img.data;
    //    console.log(`pixel values: ${pxValues}`);

    pxValues = Float32Array.from(pxValues);

    pxValues = pxValues.map(function(item){
        return item / 255.0;
    });
    //    console.log(`scaled values: ${pxValues}`);

    const X = tf.tensor([pxValues]);
    //    console.log(`Shape of Tensor:  ${X.shape}`);
    //    console.log(`dtype of Tensor: ${X.dtype}`);
    const result = model.predict(X);
    const output = result.dataSync()[0];


    // Test only
    //    const outCnv = document.createElement("CANVAS");
    //    cv.imshow(outCnv, img);
    //    document.body.appendChild(outCnv);


    // Clean Up
    img.delete();
    contours.delete();
    hierarchy.delete();
    M.delete();
    X.dispose();
    result.dispose();

    return output;
}