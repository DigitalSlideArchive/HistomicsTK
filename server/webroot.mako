<!DOCTYPE html>
<html lang="en">
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${title}</title>
    <link rel="stylesheet" href="//fonts.googleapis.com/css?family=Droid+Sans:400,700">
    <link rel="stylesheet" href="${staticRoot}/built/fontello/css/fontello.css">
    <link rel="stylesheet" href="${staticRoot}/built/fontello/css/animation.css">
    <link rel="stylesheet" href="${staticRoot}/built/girder_lib.min.css">
    <link rel="icon" type="image/png" href="${staticRoot}/img/Girder_Favicon.png">
    % for plugin in pluginCss:
    <link rel="stylesheet" href="${staticRoot}/built/plugins/${plugin}/plugin.min.css">
    % endfor
  </head>
  <body>
    <div id="g-global-info-apiroot" class="hide">${apiRoot}</div>
    <div id="g-global-info-staticroot" class="hide">${staticRoot}</div>
    <script src="${staticRoot}/built/girder_lib.min.js"></script>
    <script src="${staticRoot}/built/girder_app.min.js"></script>
    <script>
    $(function () {
      $('body').addClass('histomicstk-body')
      girder.router.enabled(false);
      girder.events.trigger('g:appload.before');
      var app = new girder.plugins.HistomicsTK.App({
        el: 'body',
        parentView: null,
        brandName: '${htkBrandName | js}',
        brandColor: '${htkBrandColor | js}',
        bannerColor: '${htkBannerColor | js}',
      });
      app.bindRoutes();
      girder.events.trigger('g:appload.after');
    });
    </script>
    % for plugin in pluginJs:
    <script src="${staticRoot}/built/plugins/${plugin}/plugin.min.js"></script>
    % endfor
  </body>
</html>
