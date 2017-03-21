import events from './events';
import ImageView from './views/body/ImageView';

import Router from './router';

function bindRoutes() {
    Router.route('', 'index', function () {
        events.trigger('g:navigateTo', ImageView, {});
    });
    return Router;
}

export default bindRoutes;
