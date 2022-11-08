package br.com.yuki.makoto.camera

import android.content.Context
import android.media.Image
import android.os.Bundle
import android.view.SurfaceView
import android.view.Window
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import br.com.yuki.makoto.camera.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File


class MainActivity : AppCompatActivity() {

    private val binding: ActivityMainBinding by lazy {
        ActivityMainBinding.inflate(layoutInflater)
    }

    private val loaderCallback by lazy {
        object : BaseLoaderCallback(applicationContext) {
            override fun onManagerConnected(status: Int) {
                when (status) {
                    LoaderCallbackInterface.SUCCESS -> {
                        binding.cameraView.enableFpsMeter()
                        binding.cameraView.enableView()

                        lifecycleScope.launch(Dispatchers.IO) {
                            runCatching {
                                val resource = resources.openRawResource(R.raw.haarcascade_eye)
                                val file = File(applicationContext.getDir("cascade", Context.MODE_PRIVATE), "haarcascade_eye.xml")

                                resource.use {
                                    it.copyTo(file.outputStream())
                                }

                                return@runCatching CascadeClassifier(file.absolutePath)
                            }.onSuccess {
                                withContext(Dispatchers.Main) { classifier = it }
                            }.onFailure {
                                it.printStackTrace()
                            }
                        }
                    }
                    else -> super.onManagerConnected(status)
                }
            }
        }
    }

    private val cameraCallback = object : CameraBridgeViewBase.CvCameraViewListener2 {
        private val eyeDetect by lazy {
            MatOfRect()
        }

        override fun onCameraViewStarted(width: Int, height: Int) {
        }

        override fun onCameraViewStopped() {
        }

        override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
            val gray = inputFrame.rgba()

            classifier?.apply {
                detectMultiScale(gray, eyeDetect)
                eyeDetect.toArray()
                    .sortedByDescending(Rect::area)
                    .run {
                        if (size > 2)
                            return@run subList(0, 2)
                        else
                            return@run this
                    }
                    .forEach { rect ->
                        Imgproc.rectangle(gray, rect, Scalar.all(1.0))
                    }
            }

            highpass(gray)
            return gray
        }

        fun highpass(src: Mat) {
            Imgproc.Laplacian(src, src, CvType.CV_8U, 3, 0.5, 0.0)
            src.convertTo(src, CvType.CV_8UC4)
        }

    }

    private var classifier: CascadeClassifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(binding.root)

        binding.cameraView.visibility = SurfaceView.VISIBLE
        binding.cameraView.setCvCameraViewListener(cameraCallback)
        binding.cameraView.setCameraPermissionGranted()
        binding.cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK)
    }

    override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initDebug())
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, loaderCallback)
        else
            loaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
    }

    override fun onPause() {
        super.onPause()
        binding.cameraView.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        binding.cameraView.disableView()
    }

}