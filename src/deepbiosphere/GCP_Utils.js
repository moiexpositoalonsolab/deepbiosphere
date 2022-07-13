function myFunction() {
  // get spreadsheet for joshua tree URLs
  var s=SpreadsheetApp.getActive().getSheetByName('Joshua tree images');
  var c=s.getActiveCell();
  // replace 'id' with the key from the folder url
  var fldr=DriveApp.getFolderById("<folder ID>");
  var files=fldr.getFiles();
  var names = {};
  // add all files to dictionary where key is filename and value is URL of file
  while (files.hasNext()) {
    f=files.next();
    var fname = f.getName().split('_');
    var fname = fname[fname.length-1].split('.')[0];
    Logger.log(fname)
    str='=hyperlink("' + f.getUrl() + '","' + fname + '")';
    names[fname] = str
  }
  for (var k in names){
    s.getRange(k).setFormula(names[k])
  }
}

