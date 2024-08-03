using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Net;
using System.Threading;
using System.Net.Sockets;
using System.Linq;
using System;
using System.Buffers.Binary;
using System.IO;

// UDP通信で画像を受信するクラス
public class UDPClient : MonoBehaviour
{
    [SerializeField] private RawImage background;
    const int PORT = 12345; // 受信用のポート番号
    static UdpClient udp;
    static IPEndPoint remoteEP;
    Thread thread;
    private Texture2D previewTexture;
    private bool imageReady = false;
    private byte[] imageBytes;
    // Start is called before the first frame update
    void Start()
    {
        udp = new UdpClient(PORT);
        // udp.Client.ReceiveTimeout = 1000;
        remoteEP = new IPEndPoint(IPAddress.Any, 0);
        thread = new Thread(new ThreadStart(ThreadMethod));
        thread.Start();
    }

    private void ThreadMethod()
    {
        Dictionary<int, byte[]> receivedPackets = new Dictionary<int, byte[]>();
        while (true)
        {
            byte[] receivedBytes = udp.Receive(ref remoteEP);
            if (receivedBytes.Length < 10)
            {
                Debug.Log("Invalid packet received");
                continue;
            }
            // ヘッダー解析 (4バイトの現在のチャンク番号, 4バイトの合計チャンク数, 1バイトの最後のチャンクフラグ)
            int currentChunk = BinaryPrimitives.ReadInt32BigEndian(new ReadOnlySpan<byte>(receivedBytes, 0, 4)); // 0-3バイト
            int totalChunks = BinaryPrimitives.ReadInt32BigEndian(new ReadOnlySpan<byte>(receivedBytes, 4, 4)); // 4-7バイト
            bool lastChunkReceived = receivedBytes[8] == 1; // 8バイト

            // ヘッダーを除いたデータ部分を取得
            byte[] data = receivedBytes.Skip(9).ToArray();
            receivedPackets[currentChunk] = data;
            currentChunk++;
            Debug.Log("Chunk " + currentChunk + "/" + totalChunks + " received");
            if (lastChunkReceived)
            {
                Debug.Log("Last chunk received");
                // 全てのパケットを受信したか確認
                if (receivedPackets.Count == totalChunks)
                {
                    Debug.Log("All chunks received");
                    // 受信したパケットを結合
                    var imageData = new List<byte>();
                    for (int i = 0; i < totalChunks; i++)
                    {
                        imageData.AddRange(receivedPackets[i]);
                    }
                    imageBytes = imageData.ToArray();
                    imageReady = true;
                    receivedPackets.Clear(); // 次の画像のためにクリア
                }else
                {
                    Debug.Log("Not all chunks received");
                    receivedPackets.Clear(); // 次の画像のためにクリア
                }
            }
        }
    }

    void Update()
    {
        if (imageReady)
        {
            if (previewTexture == null)
            {
                previewTexture = new Texture2D(2, 2);
            }
            // Debug.Log("Image received");
            // imageBytesを画像として保存
            if(previewTexture.LoadImage(imageBytes))
            {
                Debug.Log("Image loaded");
                // デバッグ用: バイトデータをファイルに書き出す
                // File.WriteAllBytes("ReceivedImage.png", imageBytes);
            }else
            {
                Debug.Log("Image size: " + imageBytes.Length);
                Debug.Log("Image load failed");
            }
            background.texture = previewTexture;
            imageReady = false;
            Resources.UnloadUnusedAssets();
            System.GC.Collect();
        }
    }

    // 終了時にスレッドを終了
    void OnApplicationQuit()
    {
        if (udp != null)
        {
            udp.Close();
        }
        if (thread != null && thread.IsAlive)
        {
            thread.Abort();
        }
    }
}
