using UnityEngine;
using System;
using Renci.SshNet;
using UnityEngine.UI;
using TMPro;
using System.Threading;
using System.Text;
public class SSHConnector : MonoBehaviour
{
    // �C���X�y�N�^�[����ݒ�
    [SerializeField] private string host_name;
    [SerializeField] private string user_name;
    [SerializeField] private string password;
    [SerializeField] private TextMeshProUGUI text;
    [SerializeField] private TextMeshProUGUI output_log;
    [SerializeField] private Button button;
    private int port = 22;
    private SshClient ssh;
    public static ShellStream shellStream;
    private int max_lines = 10;
    StringBuilder output;
    // Start is called before the first frame update
    void Start()
    {
        text.text = "SSH";
        if (String.IsNullOrEmpty(host_name) || String.IsNullOrEmpty(user_name) || String.IsNullOrEmpty(password))
        {
            Debug.LogError("Please set host_name, user_name, password in the inspector.");
            return;
        }
        button.onClick.AddListener(OnClick);
        output = new StringBuilder();
    }

    public void OnClick()
    {
        text.text = "Connecting...";
        try
        {
            ConnectionInfo info = new ConnectionInfo(host_name, port, user_name, new PasswordAuthenticationMethod(user_name, password));
            // SSH�ڑ�
            ssh = new SshClient(info);
            ssh.Connect();
            if (ssh.IsConnected)
            {
                Debug.Log("SSH connection established.");
                text.text = "Connected";
                if (shellStream == null)
                {
                    shellStream = ssh.CreateShellStream("xterm", 80, 24, 800, 600, 1024);
                }
            }
            else
            {
                Debug.LogError("SSH connection failed.");
                text.text = "Failed";
                return;
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error: " + e.Message);
            text.text = "Failed";
            return;
        }
    }

    void Update()
    {
        if (shellStream != null)
        {
            string line = shellStream.ReadLine(TimeSpan.FromSeconds(0.5));
            // line��null��������C\n�݂̂̏ꍇ�͉������Ȃ�
            if (line == null || line == "\n")
            {
                return;
            }
            output.AppendLine(line);
            if (output.ToString().Split('\n').Length > max_lines)
            {
                output.Remove(0, output.ToString().IndexOf('\n') + 1);
            }
            output_log.text = output.ToString();
        }
        else
        {
            text.text = "SSH";
        }
    }
}
