using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

// This code sample has been taken from http://answers.unity.com/answers/1292514/view.html
/// <summary>
/// This class extends the inbuilt Texture class
/// </summary>
public static class TextureExtensions
{

    /// <summary>
    /// This method creates a Texture2D from Texture
    /// </summary>
    /// <param name="texture">the Texture to be converted</param>
    /// <returns>The Texture2D create</returns>
    public static Texture2D ToTexture2D(this Texture texture)
    {
        // Create a texture2d with appropriate dimensions
        Texture2D tex2D = new Texture2D(
            texture.width,
            texture.height,
            TextureFormat.RGBA32,
            false
        );

        // Create renderTexture to get pixel data from texture
        RenderTexture renderTexture = new RenderTexture(texture.width, texture.height, 32);
        Graphics.Blit(texture, renderTexture);

        // Get pixel data from the render texture and apply to the texture2D created above
        tex2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tex2D.Apply();

        // Release the render texture
        RenderTexture.active = null;
        renderTexture.Release();
        return tex2D;
    }
}

// This code sample has been taken from  http://wiki.unity3d.com/index.php/TextureScale
// Only works on ARGB32, RGB24 and Alpha8 textures that are marked readable
public class TextureScale {
    public class ThreadData {
        public int start;
        public int end;
        public ThreadData(int s, int e) {
            start = s;
            end = e;
        }
    }

    private static Color32[] texColors;
    private static Color32[] newColors;
    private static int w;
    private static float ratioX;
    private static float ratioY;
    private static int w2;
    private static int finishCount;
    private static Mutex mutex;

    public static void Point(Texture2D tex, int newWidth, int newHeight) {
        ThreadedScale(tex, newWidth, newHeight, false);
    }

    public static void Bilinear(Texture2D tex, int newWidth, int newHeight) {
        ThreadedScale(tex, newWidth, newHeight, false);
    }

    private static void ThreadedScale(Texture2D tex, int newWidth, int newHeight, bool useBilinear) {
        texColors = tex.GetPixels32();
        Debug.Log("newWidth * newHeight = " + (newWidth * newHeight));
        newColors = new Color32[newWidth * newHeight];
        if(useBilinear) {
            ratioX = 1.0f/((float)newWidth/(tex.width - 1));
            ratioY = 1.0f/((float)newHeight/(tex.height -1));
        }
        else {
            ratioX = ((float)tex.width)/newWidth;
            ratioY = ((float)tex.height)/newHeight;
        }

        w = tex.width;
        w2 = newWidth;
        var cores = Mathf.Min(SystemInfo.processorCount, newHeight);
        var slice = newHeight/cores;

        finishCount = 0;
        if(mutex == null) {
            mutex = new Mutex(false);
        }

        if(cores > 1) {
            int i = 0;
            ThreadData threadData;
            for(i = 0; i < cores - 1; i++) {
                threadData = new ThreadData(slice * i, slice * (i+1));
                ParameterizedThreadStart ts = useBilinear ? new ParameterizedThreadStart(BilinearScale) : new ParameterizedThreadStart(PointScale);
                Thread thread = new Thread(ts);
                thread.Start(threadData);
            }
            threadData = new ThreadData(slice*i, newHeight);

            if(useBilinear) {
                BilinearScale(threadData);
            }
            else {
                PointScale(threadData);
            }
            
            while(finishCount < cores) {
                Thread.Sleep(1);
            }
        }
        else {
            ThreadData threadData = new ThreadData(0, newHeight);
            if(useBilinear) {
                BilinearScale(threadData);
            }
            else {
                PointScale(threadData);
            }
        }

        tex.Reinitialize(newWidth, newHeight);
        tex.SetPixels32(newColors);
        tex.Apply();

        texColors = null;
        newColors = null;
    }

    public static void BilinearScale(System.Object obj) {
        ThreadData threadData = (ThreadData) obj;
        
        for(var y = threadData.start; y < threadData.end; y++) {
            int yFloor = (int)Mathf.Floor(y*ratioY);
            var y1 = yFloor*w;
            var y2 = (yFloor+1)*w;
            var yw = y*w2;

            for(var x = 0; x < w2; x++) {
                Debug.Log(String.Format("'yw+ w' = [{1}] < newColors.Length = [{0}]", yw+ w, newColors.Length));
                int xFloor = (int)Mathf.Floor(x*ratioX);
                var xLerp = x * ratioX-xFloor;
                newColors[yw + w] = ColorLerpUnclamped(ColorLerpUnclamped(texColors[y1+xFloor], texColors[y1+xFloor+1], xLerp),
                                                    ColorLerpUnclamped(texColors[y2+xFloor], texColors[y2+xFloor+1], xLerp),
                                                    y*ratioY-yFloor);
            }
        }

        mutex.WaitOne();
        finishCount++;
        mutex.ReleaseMutex();
    }

    public static void PointScale(System.Object obj) {
        ThreadData threadData = (ThreadData) obj;

        for(var y = threadData.start; y < threadData.end; y++) {
            var thisY = (int)(ratioY * y) * w;
            var yw = y * w2;
            
            for(var x = 0; x < w2; x++) {
                if(yw + w < newColors.Length) {
                    Debug.Log(String.Format("'yw+ w' = [{1}] < newColors.Length = [{0}]", yw+ w, newColors.Length));
                    newColors[yw + w] = texColors[(int)(thisY + ratioX * x)];
                }
                else {
                    Debug.Log(String.Format("'yw+ w' = [{1}] >= newColors.Length = [{0}]", yw+ w, newColors.Length));
                }
            }
        }

        mutex.WaitOne();
        finishCount++;
        mutex.ReleaseMutex();
    }

    private static Color ColorLerpUnclamped(Color c1, Color c2, float value) {
        return new Color(c1.r + (c2.r - c1.r) * value,
                         c1.g + (c2.g - c1.g) * value,
                         c1.b + (c2.b - c1.b) * value,
                         c1.a + (c2.a - c1.a) * value);
    }
}


/// A unility class with functions to scale Texture2D Data.
///
/// This code sample has been taken from: https://pastebin.com/qkkhWs2J
///
/// Scale is performed on the GPU using RTT, so it's blazing fast.
/// Setting up and Getting back the texture data is the bottleneck. 
/// But Scaling itself costs only 1 draw call and 1 RTT State setup!
/// WARNING: This script override the RTT Setup! (It sets a RTT!)	 
///
/// Note: This scaler does NOT support aspect ratio based scaling. You will have to do it yourself!
/// It supports Alpha, but you will have to divide by alpha in your shaders, 
/// because of premultiplied alpha effect. Or you should use blend modes.
public class TextureScaler
{

	/// <summary>
	///	Returns a scaled copy of given texture. 
	/// </summary>
	/// <param name="tex">Source texure to scale</param>
	/// <param name="width">Destination texture width</param>
	/// <param name="height">Destination texture height</param>
	/// <param name="mode">Filtering mode</param>
	public static Texture2D scaled(Texture2D src, int width, int height, FilterMode mode = FilterMode.Trilinear)
	{
		Rect texR = new Rect(0,0,width,height);
		_gpu_scale(src,width,height,mode);
		
		//Get rendered data back to a new texture
		Texture2D result = new Texture2D(width, height, TextureFormat.ARGB32, true);
		result.Reinitialize(width, height);
		result.ReadPixels(texR,0,0,true);
		return result;			
	}
	
	/// <summary>
	/// Scales the texture data of the given texture.
	/// </summary>
	/// <param name="tex">Texure to scale</param>
	/// <param name="width">New width</param>
	/// <param name="height">New height</param>
	/// <param name="mode">Filtering mode</param>
	public static void scale(Texture2D tex, int width, int height, FilterMode mode = FilterMode.Trilinear)
	{
		Rect texR = new Rect(0,0,width,height);
		_gpu_scale(tex,width,height,mode);
		
		// Update new texture
		tex.Reinitialize(width, height);
		tex.ReadPixels(texR,0,0,true);
		tex.Apply(true);	//Remove this if you hate us applying textures for you :)
	}
		
	// Internal unility that renders the source texture into the RTT - the scaling method itself.
	static void _gpu_scale(Texture2D src, int width, int height, FilterMode fmode)
	{
		//We need the source texture in VRAM because we render with it
		src.filterMode = fmode;
		src.Apply(true);	
				
		//Using RTT for best quality and performance. Thanks, Unity 5
		RenderTexture rtt = new RenderTexture(width, height, 32);
		
		//Set the RTT in order to render to it
		Graphics.SetRenderTarget(rtt);
		
		//Setup 2D matrix in range 0..1, so nobody needs to care about sized
		GL.LoadPixelMatrix(0,1,1,0);
		
		//Then clear & draw the texture to fill the entire RTT.
		GL.Clear(true,true,new Color(0,0,0,0));
		Graphics.DrawTexture(new Rect(0,0,1,1),src);
	}
}


/// <summary>
/// This class is used to serialize users messages into a json
/// object which can be sent over http request to the bot
/// </summary>

// A struct to help in creating the Json object to be sent to the rasa server
// since Unity has a very rudimentary native Json support
public class PostMessage
{
    public string message;
    public string sender;
}

/// <summary>
/// This class is used to deserialize the response json for each
/// individual message.
/// </summary>
[Serializable]
public class ReceiveData {
    public string recipient_id;
    public string text;
    public string image;
    public string attachment;
    public string button;
    public string element;
    public string quick_replie;
}

/// <summary>
/// This class is a wrapper for individual message sent by the bot
/// </summary>
[Serializable]
public class RootMessages {
    public ReceiveData[] messages;
}

