var plugin = {

   GetExternalJS: function () {

      var onnx_script = document.createElement("script");
      onnx_script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
      document.head.appendChild(onnx_script);

      var script = document.createElement("script");
      script.src = "./StreamingAssets/main.js";
      document.head.appendChild(script);
   },

   InitOrtSession: async function (model_path, exec_provider, mean, std_dev) {

      // ort.env.wasm.numThreads = 8;

      // Convert bytes to the text
      var model_path_str = UTF8ToString(model_path);
      var exec_provider_str = UTF8ToString(exec_provider);

      const sessionOption = {
         executionProviders: [exec_provider_str],
         graphOptimizationLevel: 'all',
         // intraOpNumThreads: 8,
         // interOpNumThreads: 1,
         enableCpuMemArena: false,
         enableMemPattern: false,
         executionMode: 'sequential',
      };

      this.session = await ort.InferenceSession.create(model_path_str, sessionOption);

      this.mean = new Float32Array(buffer, mean, 3);
      this.std_dev = new Float32Array(buffer, std_dev, 3);
      console.log(`Input Name: ${this.session.inputNames[0]}`);
      console.log(`Output Name: ${this.session.outputNames[0]}`);
   },

   PerformInference: function (array_data, size, width, height) {
      if (typeof this.session == 'undefined') {
         console.log(`Session not defined yet (PerformInference)`);
         return;
      }

      // 
      const uintArray = new Uint8ClampedArray(buffer, array_data, size, width, height);
      uintArray.reverse();
      // 
      const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());

      // 
      for (let i = 0; i < uintArray.length; i += 3) {
         redArray.push(((uintArray[i + 2] / 255.0) - this.mean[0]) / this.std_dev[0]);
         greenArray.push(((uintArray[i + 1] / 255.0) - this.mean[1]) / this.std_dev[1]);
         blueArray.push(((uintArray[i] / 255.0) - this.mean[2]) / this.std_dev[2]);
         // redArray.push(((uintArray[i] / 255.0) - this.mean[0]) / this.std_dev[0]);
         // greenArray.push(((uintArray[i + 1] / 255.0) - this.mean[1]) / this.std_dev[1]);
         // blueArray.push(((uintArray[i + 2] / 255.0) - this.mean[2]) / this.std_dev[2]);
      }

      // 
      const input_data = Float32Array.from(redArray.concat(greenArray).concat(blueArray));

      // 
      const input_tensor = new ort.Tensor("float32", input_data, [1, 3, height, width]);
      const feeds = {};
      feeds[this.session.inputNames[0]] = input_tensor;

      // 
      PerformInferenceAsync(session, feeds).then(outputData => {
         const output = outputData[session.outputNames[0]];
         this.index = argMax(Array.prototype.slice.call(output.data));
      })
      return this.index;
   },
}

// Creating functions for the Unity
mergeInto(LibraryManager.library, plugin);