import events from './events';
import FrontPageView from './views/body/FrontPageView';
import ImageView from './views/body/ImageView';

import Router from './router';

function bindRoutes() {
    Router.route('', 'index', function () {
        events.trigger('g:navigateTo', FrontPageView);
    });

    Router.route('image', 'image', function () {
        events.trigger('g:navigateTo', ImageView, {});
    });
    return Router;
}

export default bindRoutes;
