var myDate = new Date();

// var session = ort.InferenceSession;


function displayDate() {
    window.alert(myDate);
}

function printFloatArray(array, length) {

    console.log(array);

    for (let i = 0; i < length; i++) {
        array[i] *= 2;
    }
}

async function PerformInferenceAsync(session, feeds) {

    const outputData = await session.run(feeds);
    return outputData;
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

// async function init_session(model_path, exec_provider) {
//     var return_msg;
//     try {
//         // create a new session and load the specified model.
//         session = await ort.InferenceSession.create(model_path,
//             { executionProviders: [exec_provider], graphOptimizationLevel: 'all' });
//         return_msg = 'Created inference session.';
//     } catch (e) {
//         return_msg = `failed to create inference session: ${e}.`;
//     }
//     console.log(return_msg);
// }