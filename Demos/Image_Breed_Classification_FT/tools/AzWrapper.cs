using System;
using System.Diagnostics;

internal static class AzWrapper
{
    private static int Main(string[] args)
    {
        var command = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("ComSpec") ?? "cmd.exe",
            Arguments = "/d /s /c az.cmd " + string.Join(" ", Array.ConvertAll(args, Quote)),
            UseShellExecute = false
        };

        using (var process = Process.Start(command))
        {
            process.WaitForExit();
            return process.ExitCode;
        }
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }
}
