import events from './events';
import FrontPageView from './views/body/FrontPageView';

import Router from './router';

Router.route('', 'index', function () {
    events.trigger('g:navigateTo', FrontPageView);
});
