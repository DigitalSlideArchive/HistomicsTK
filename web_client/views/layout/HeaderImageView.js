import events from '../../events';
import View from '../View';

import headerImageTemplate from '../../templates/layout/headerImage.pug';
import '../../stylesheets/layout/headerImage.styl';

var HeaderImageView = View.extend({
    events: {
        'click .h-open-image': function (evt) {
            events.trigger('h:openImageUi');
        },
        'click .h-open-annotated-image': function (evt) {
            events.trigger('h:openAnnotatedImageUi');
        }
    },

    initialize() {
        this.imageModel = null;
        this.parentChain = null;
        this.listenTo(events, 'h:imageOpened', (model) => {
            this.imageModel = model;
            this.parentChain = null;
            if (model) {
                this.imageModel.getRootPath((resp) => {
                    this.parentChain = resp;
                    this.render();
                });
            }
            this.render();
        });
    },

    render() {
        this.$el.html(headerImageTemplate({
            image: this.imageModel,
            parentChain: this.parentChain
        }));
        return this;
    }
});

export default HeaderImageView;
