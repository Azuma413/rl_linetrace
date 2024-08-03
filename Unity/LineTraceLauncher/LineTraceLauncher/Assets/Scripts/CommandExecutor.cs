using UnityEngine;
using UnityEngine.UI;
using System;
using TMPro;

public class CommandExecutor : MonoBehaviour
{
    [SerializeField] private string command;
    [SerializeField] private Button button;
    [SerializeField] private TextMeshProUGUI text;
    private bool flag = false;

    void Start()
    {
        if (string.IsNullOrEmpty(command))
        {
            Debug.LogError("Please set command in the inspector.");
            return;
        }
        button.onClick.AddListener(ExecuteCommand);
        text.text = "Execute";
    }

    public void ExecuteCommand()
    {
        try{
            if (SSHConnector.shellStream != null){
                // �R�}���h�̎��s
                if (flag)
                {
                    SSHConnector.shellStream.Write("\x3"); // Ctrl+C�𑗐M
                    SSHConnector.shellStream.Write("exit\n"); // exit�𑗐M
                    SSHConnector.shellStream.Close(); // shellStream�����
                    SSHConnector.shellStream.Dispose(); // shellStream��j��
                    SSHConnector.shellStream = null;
                    flag = false;
                    text.text = "Execute";
                    return;
                }
                else
                {
                    SSHConnector.shellStream.WriteLine(command);
                    flag = true;
                    text.text = "Stop";
                    Debug.LogFormat("[CMD] {0}", command);
                }
            }
            else
            {
                Debug.LogError("Shell stream is not available.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error: " + e.Message);
        }
    }

}
