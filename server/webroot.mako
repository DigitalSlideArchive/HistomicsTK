<!DOCTYPE html>
<html lang="en">
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${title}</title>
    <link rel="stylesheet" href="//fonts.googleapis.com/css?family=Droid+Sans:400,700">
    <link rel="stylesheet" href="${staticRoot}/built/fontello/css/fontello.css">
    <link rel="stylesheet" href="${staticRoot}/built/fontello/css/animation.css">
    <link rel="stylesheet" href="${staticRoot}/built/girder.ext.min.css">
    <link rel="stylesheet" href="${staticRoot}/built/girder.app.min.css">
    <link rel="icon" type="image/png" href="${staticRoot}/img/Girder_Favicon.png">
    % for plugin in pluginCss:
    <link rel="stylesheet" href="${staticRoot}/built/plugins/${plugin}/plugin.min.css">
    % endfor
  </head>
  <body>
    <div id="g-global-info-apiroot" class="hide">${apiRoot}</div>
    <div id="g-global-info-staticroot" class="hide">${staticRoot}</div>
    <script src="${staticRoot}/built/girder.ext.min.js"></script>
    <script src="${staticRoot}/built/girder.app.min.js"></script>
    % for plugin in pluginJs:
    <script src="${staticRoot}/built/plugins/${plugin}/plugin.min.js"></script>
    % endfor
    <script src="${staticRoot}/built/plugins/HistomicsTK/histomics.main.min.js"></script>
  </body>
</html>
