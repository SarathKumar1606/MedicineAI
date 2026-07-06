<!DOCTYPE html>
<html>
<head>
<script language="JavaScript">
flag=1;
function addemail()
{

let a=document.f1.email.value;
if(a.indexOf("@")!=a.lastIndexOf("@"))
flag=0;
else if(a.indexOf(".")<a.indexOf("@"))
flag=0;
else if(a.indexOf(".")-1==a.indexOf("@"))
flag=0;
else if(a.indexOf("@")==0)
flag=0;
else if(a.length-4!=a.indexOf("."))
flag=0;


for(i=0;i<a.length;i++)
{
    b=(a.charCodeAt(i)>=97 && a.charCodeAt(i)<=126) || (a.charCodeAt(i) == 64) || (a.charCodeAt(i)==46) || (a.charCodeAt(i)>=48 && a.charCodeAt(i) <=58)
    if(b!= true)
    {
      flag=0;
        break;
      }
   }


if (flag==0)
document.write("invalid");
else
document.write("valid");
}




</script>
</head>
<body>
<form id="f1" name="f1">

<label>EMAIL</Label>
<input type="email" name="email"/>
<input type="button" value="Chk" onClick="addemail()"/>
</form>
</body>
</html>